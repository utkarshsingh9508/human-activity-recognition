"""Model factory. `arch` and `feat` are plain dicts stored in every checkpoint,
so any saved model can be rebuilt exactly."""
from features import layout_of, num_channels

from .cnn_lstm import CNNLSTM
from .graph import build_adjacency
from .stgcn import STGCN

MODEL_NAMES = ["stgcn", "stgcn_lite", "agcn", "cnn_lstm"]


def build_model(arch, feat, num_classes):
    name = arch["name"]
    C = num_channels(feat)
    lay = layout_of(feat)
    if name == "cnn_lstm":
        return CNNLSTM(num_classes, in_channels=C, num_nodes=lay.num_nodes)
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}, choose from {MODEL_NAMES}")
    A = build_adjacency(lay.num_nodes, lay.edges, lay.center, arch.get("partition", "spatial"))
    return STGCN(num_classes, A, C,
                 width=32 if name == "stgcn_lite" else 64,
                 adaptive=(name == "agcn"),
                 edge_importance=arch.get("edge_importance", True))
