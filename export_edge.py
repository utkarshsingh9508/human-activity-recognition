"""Edge-AI export: PyTorch -> ONNX FP32 -> ONNX INT8 (static quantisation),
then benchmark all three on the test split.

    python export_edge.py --ckpt checkpoints/stgcn.pt
    python export_edge.py --ckpt checkpoints/stgcn_bone.pt
"""
import argparse
import inspect
import json

import numpy as np

import utils  # noqa: F401
import torch

import config as C
from dataset import SkeletonDataset, load_split
from evaluate import evaluate, table
from inference import Predictor, load_checkpoint


def export_onnx(model, meta, shape, path):
    """Classic TorchScript exporter first; newer PyTorch versions may only offer the
    dynamo exporter (needs the `onnxscript` package), so fall back to it."""
    args = (torch.randn(1, *shape),)
    common = dict(input_names=["skeleton"], output_names=["logits"],
                  dynamic_axes={"skeleton": {0: "batch"}, "logits": {0: "batch"}})
    has_dynamo = "dynamo" in inspect.signature(torch.onnx.export).parameters
    try:
        kwargs = {"dynamo": False} if has_dynamo else {}
        torch.onnx.export(model, args, str(path), opset_version=17, **common, **kwargs)
    except Exception as e:
        if not has_dynamo:
            raise
        print(f"classic ONNX exporter failed ({e.__class__.__name__}); using dynamo exporter")
        torch.onnx.export(model, args, str(path), dynamo=True, opset_version=18, **common)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))


def quantize_int8(fp32, int8, meta, n_calib):
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    split = load_split(C.ROOT / meta["split_file"])
    ds = SkeletonDataset(split["train"], meta["profile"], meta["features"], meta["seq_len"])
    idx = np.random.default_rng(0).choice(len(ds), size=min(n_calib, len(ds)), replace=False)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"skeleton": ds[int(i)][0].numpy()[None]} for i in idx])

        def get_next(self):
            return next(self.it, None)

    prep = fp32.with_name(fp32.stem + "_prep.onnx")
    quant_pre_process(str(fp32), str(prep))
    quantize_static(str(prep), str(int8), Reader(), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8)
    prep.unlink(missing_ok=True)
    int8.with_suffix(".json").write_text(json.dumps(meta, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--calib", type=int, default=300)
    ap.add_argument("--no-int8", action="store_true")
    args = ap.parse_args()

    model, meta = load_checkpoint(args.ckpt, "cpu")
    shape = Predictor(args.ckpt, "cpu").input_shape()
    C.EXPORT_DIR.mkdir(exist_ok=True)
    fp32 = C.EXPORT_DIR / f"{meta['tag']}_fp32.onnx"
    int8 = C.EXPORT_DIR / f"{meta['tag']}_int8.onnx"

    export_onnx(model, meta, shape, fp32)
    x = np.random.randn(8, *shape).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    diff = float(np.abs(ref - Predictor(fp32).logits(x)).max())
    print(f"exported {fp32}  max |PyTorch - ONNX| = {diff:.2e} ({'OK' if diff < 1e-3 else 'CHECK'})")

    files = [args.ckpt, fp32]
    if not args.no_int8:
        try:
            quantize_int8(fp32, int8, meta, args.calib)
            print(f"quantised {int8}")
            files.append(int8)
        except Exception as e:
            print(f"INT8 quantisation skipped: {e}")

    rows = [evaluate([f], "test", verbose=False) for f in files]
    t = table(rows)
    print("\n" + t)
    (C.RESULTS_DIR / f"edge_benchmark_{meta['tag']}.md").write_text(t + "\n")


if __name__ == "__main__":
    main()
