"""CNN-LSTM baseline (treats the skeleton as a flat vector, ignores the graph).
Used to show how much the graph structure of ST-GCN helps."""
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, num_classes, in_channels=7, num_nodes=35, hidden=128, dropout=0.3):
        super().__init__()
        self.feat = in_channels * num_nodes
        self.input_bn = nn.BatchNorm1d(self.feat)
        self.cnn = nn.Sequential(
            nn.Conv1d(self.feat, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(128, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(inplace=True),
                                nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x):
        t = x.shape[2]
        x = x.permute(0, 1, 3, 2).reshape(-1, self.feat, t)
        x = self.cnn(self.input_bn(x))
        out, _ = self.lstm(x.transpose(1, 2))
        return self.fc(out[:, -1])
