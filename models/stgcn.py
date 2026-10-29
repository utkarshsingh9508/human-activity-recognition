"""ST-GCN and its adaptive-graph variant (AGCN).

Input (N, C, T, V) -> logits (N, num_classes).

Graph convolution per partition k:   X * W_k * A_k
  edge importance (ST-GCN):          A_k * M_k           M_k learnable, init 1
  adaptive graph  (2s-AGCN):         + B_k               learnable global links, init 0
                                     + alpha * C_k(X)    data-dependent links (attention
                                                         between joints for THIS sample)
B and C let the network connect joints that are not bones, e.g. both hands when clapping.
Plain matmul/softmax ops only, so the model exports to ONNX and INT8."""
import torch
import torch.nn as nn


class GraphConv(nn.Module):
    def __init__(self, in_c, out_c, K, V, adaptive=False):
        super().__init__()
        self.K, self.out_c, self.adaptive = K, out_c, adaptive
        self.conv = nn.Conv2d(in_c, out_c * K, kernel_size=1)
        if adaptive:
            self.inter = max(out_c // 4, 8)
            self.B = nn.Parameter(torch.zeros(K, V, V))
            self.theta = nn.Conv2d(in_c, self.inter * K, 1)
            self.phi = nn.Conv2d(in_c, self.inter * K, 1)
            self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, A):
        t, v = x.shape[2], x.shape[3]
        y = self.conv(x).view(-1, self.K, self.out_c, t, v)
        if self.adaptive:
            th = self.theta(x).view(-1, self.K, self.inter, t, v)
            th = th.permute(0, 1, 4, 2, 3).reshape(-1, self.K, v, self.inter * t)
            ph = self.phi(x).view(-1, self.K, self.inter * t, v)
            C = torch.softmax(torch.matmul(th, ph) / (self.inter * t), dim=-2)   # (N,K,V,V)
        out = None
        for k in range(self.K):
            Ak = A[k]
            if self.adaptive:
                Ak = (Ak + self.B[k]) + self.alpha * C[:, k].unsqueeze(1)       # (N,1,V,V)
            term = torch.matmul(y[:, k], Ak)
            out = term if out is None else out + term
        return out


class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, K, V, stride=1, residual=True, dropout=0.25,
                 t_kernel=9, adaptive=False):
        super().__init__()
        self.gcn = GraphConv(in_c, out_c, K, V, adaptive)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, (t_kernel, 1), (stride, 1), ((t_kernel - 1) // 2, 0)),
            nn.BatchNorm2d(out_c),
            nn.Dropout(dropout),
        )
        if not residual:
            self.res = None
        elif in_c == out_c and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(in_c, out_c, 1, (stride, 1)), nn.BatchNorm2d(out_c))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        y = self.tcn(self.gcn(x, A))
        if self.res is not None:
            y = y + self.res(x)
        return self.relu(y)


class STGCN(nn.Module):
    def __init__(self, num_classes, A, in_channels, width=64, adaptive=False,
                 edge_importance=True, dropout=0.25):
        super().__init__()
        A = torch.as_tensor(A, dtype=torch.float32)
        self.register_buffer("A", A)
        K, V = A.shape[0], A.shape[1]
        self.in_channels, self.num_nodes = in_channels, V
        w = width
        cfg = [(in_channels, w, 1, False), (w, w, 1, True), (w, w, 1, True),
               (w, 2 * w, 2, True), (2 * w, 2 * w, 1, True),
               (2 * w, 4 * w, 2, True), (4 * w, 4 * w, 1, True)]
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        self.blocks = nn.ModuleList([STGCNBlock(i, o, K, V, s, r, dropout, adaptive=adaptive)
                                     for i, o, s, r in cfg])
        if edge_importance:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones_like(A)) for _ in cfg])
        else:
            self.edge_importance = None
        self.fc = nn.Linear(4 * w, num_classes)

    def forward(self, x):
        c, t, v = self.in_channels, x.shape[2], self.num_nodes
        x = x.permute(0, 3, 1, 2).reshape(-1, v * c, t)
        x = self.data_bn(x)
        x = x.reshape(-1, v, c, t).permute(0, 2, 3, 1).contiguous()
        for i, block in enumerate(self.blocks):
            A = self.A * self.edge_importance[i] if self.edge_importance is not None else self.A
            x = block(x, A)
        return self.fc(x.mean(dim=(2, 3)))
