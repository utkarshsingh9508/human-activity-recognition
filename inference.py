"""Predictors for .pt (PyTorch) and .onnx (ONNX Runtime) models, plus an Ensemble
that averages several models (e.g. joint stream + bone stream)."""
import json
import time
from pathlib import Path

import numpy as np

import utils  # noqa: F401  (sets the MPS fallback before torch is imported)
from features import prepare

META_KEYS = ("tag", "profile", "classes", "seq_len", "features", "arch", "split_file")


def load_checkpoint(path, device="cpu"):
    import torch
    from models import build_model
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    meta = {k: ckpt[k] for k in META_KEYS}
    model = build_model(meta["arch"], meta["features"], len(meta["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), meta


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Predictor:
    def __init__(self, path, device="auto", threads=None):
        self.path = Path(path)
        if self.path.suffix == ".onnx":
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if threads:
                opts.intra_op_num_threads = threads
            self.sess = ort.InferenceSession(str(self.path), opts, providers=["CPUExecutionProvider"])
            self.input_name = self.sess.get_inputs()[0].name
            self.meta = json.loads(self.path.with_suffix(".json").read_text())
            self.backend = "onnxruntime-cpu"
        else:
            import torch
            if threads:
                torch.set_num_threads(threads)
            self.device = utils.get_device(device)
            self.model, self.meta = load_checkpoint(self.path, self.device)
            self.backend = f"pytorch-{self.device}"
        self.classes = self.meta["classes"]
        self.seq_len = self.meta["seq_len"]
        self.features = self.meta["features"]
        self.name = self.meta["tag"]

    def preprocess(self, raw_window):
        """Raw (T, J, 4) keypoints (last seq_len frames used) -> (1, C, T, V)."""
        return prepare(raw_window[-self.seq_len:], self.features)[None]

    def logits(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if hasattr(self, "sess"):
            return self.sess.run(None, {self.input_name: x})[0]
        import torch
        with torch.no_grad():
            return self.model(torch.from_numpy(x).to(self.device)).float().cpu().numpy()

    def probs(self, x):
        return _softmax(self.logits(x))

    def input_shape(self):
        from features import layout_of, num_channels
        return (num_channels(self.features), self.seq_len, layout_of(self.features).num_nodes)

    def latency_ms(self, runs=200, warmup=20):
        x = np.random.randn(1, *self.input_shape()).astype(np.float32)
        for _ in range(warmup):
            self.logits(x)
        t0 = time.perf_counter()
        for _ in range(runs):
            self.logits(x)
        return (time.perf_counter() - t0) / runs * 1000

    def size_mb(self):
        return self.path.stat().st_size / 1e6


class Ensemble:
    """Average of the softmax outputs of several predictors (multi-stream fusion)."""

    def __init__(self, paths, device="auto", threads=None):
        self.members = [Predictor(p, device, threads) for p in paths]
        first = self.members[0]
        for m in self.members[1:]:
            if m.classes != first.classes:
                raise SystemExit("models to fuse must be trained on the same classes")
        self.classes = first.classes
        self.seq_len = max(m.seq_len for m in self.members)
        self.name = "+".join(m.name for m in self.members)
        self.backend = "/".join(sorted({m.backend for m in self.members}))
        self.features = first.features

    def probs_raw(self, raw_window):
        return np.mean([m.probs(m.preprocess(raw_window))[0] for m in self.members], axis=0)
