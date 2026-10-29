"""Evaluate models on the held-out split they were trained with.

    python evaluate.py --model checkpoints/stgcn.pt checkpoints/cnn_lstm.pt      # compare
    python evaluate.py --model checkpoints/stgcn.pt checkpoints/stgcn_bone.pt --fuse  # two-stream
    python evaluate.py --model exports/stgcn_fp32.onnx exports/stgcn_int8.onnx

Metrics
  window accuracy : every 30-frame window classified separately (what real-time sees)
  clip accuracy   : window scores averaged over each recording (what papers report)
  macro-F1, CPU latency per window (all fused models together), model size
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import utils  # noqa: F401
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from config import RESULTS_DIR, ROOT
from dataset import SkeletonDataset, load_split
from inference import Predictor


def plot_confusion(cm, classes, title, path):
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    n = len(classes)
    size = min(0.45 * n + 3, 30)
    fig, ax = plt.subplots(figsize=(size, size - 1))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), classes, rotation=45, ha="right", fontsize=8 if n > 15 else 10)
    ax.set_yticks(range(n), classes, fontsize=8 if n > 15 else 10)
    if n <= 15:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def predict_split(pred, split, subset):
    ds = SkeletonDataset(split[subset], split["profile"], pred.features, pred.seq_len, train=False)
    probs, ys, cids = [], [], []
    for x, y, c in DataLoader(ds, batch_size=128):
        probs.append(pred.probs(x.numpy()))
        ys.append(y.numpy())
        cids.append(c.numpy())
    return np.concatenate(probs), np.concatenate(ys), np.concatenate(cids)


def clip_accuracy(probs, y, cids):
    correct = []
    for c in np.unique(cids):
        m = cids == c
        correct.append(probs[m].mean(axis=0).argmax() == y[m][0])
    return float(np.mean(correct))


def evaluate(paths, subset="test", name=None, verbose=True):
    preds = [Predictor(p) for p in paths]
    split_file = preds[0].meta["split_file"]
    for p in preds[1:]:
        if p.meta["split_file"] != split_file or p.classes != preds[0].classes:
            raise SystemExit("models to fuse/compare together must share classes and split")
    split = load_split(ROOT / split_file)
    classes = preds[0].classes

    outs = [predict_split(p, split, subset) for p in preds]
    y, cids = outs[0][1], outs[0][2]
    probs = np.mean([o[0] for o in outs], axis=0)
    pred_y = probs.argmax(axis=1)

    win_acc = accuracy_score(y, pred_y)
    clip_acc = clip_accuracy(probs, y, cids)
    f1 = f1_score(y, pred_y, average="macro")
    labels = list(range(len(classes)))
    report = classification_report(y, pred_y, labels=labels, target_names=classes,
                                   digits=4, zero_division=0)
    latency = sum((p if p.backend.endswith("cpu") else Predictor(p.path, "cpu")).latency_ms()
                  for p in preds)
    size = sum(p.size_mb() for p in preds)

    tag = name or "+".join(p.path.stem for p in preds)
    RESULTS_DIR.mkdir(exist_ok=True)
    text = (f"Model(s): {', '.join(p.path.name for p in preds)}\n"
            f"Split: {split_file} [{subset}]  ({len(np.unique(cids))} clips, {len(y)} windows)\n"
            f"Window accuracy: {win_acc:.4f}\nClip accuracy:   {clip_acc:.4f}\n"
            f"Macro F1:        {f1:.4f}\n"
            f"CPU latency:     {latency:.2f} ms per window\nSize:            {size:.2f} MB\n\n{report}")
    (RESULTS_DIR / f"{tag}_{subset}_report.txt").write_text(text)
    plot_confusion(confusion_matrix(y, pred_y, labels=labels), classes,
                   f"{tag} [{subset}] window acc {win_acc:.2%}",
                   RESULTS_DIR / f"{tag}_{subset}_confusion.png")
    summary = {"name": tag, "window_acc": win_acc, "clip_acc": clip_acc, "macro_f1": f1,
               "latency_ms": latency, "size_mb": size, "split": split_file, "subset": subset}
    (RESULTS_DIR / f"{tag}_{subset}.json").write_text(json.dumps(summary, indent=1))
    if verbose:
        print(text)
    return summary


def table(rows):
    lines = ["| Model | Window acc | Clip acc | Macro-F1 | CPU ms / window | Size (MB) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['window_acc']:.2%} | {r['clip_acc']:.2%} | "
                     f"{r['macro_f1']:.4f} | {r['latency_ms']:.2f} | {r['size_mb']:.2f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--fuse", action="store_true", help="average the models (multi-stream)")
    ap.add_argument("--subset", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()
    if args.fuse:
        rows = [evaluate(args.model, args.subset)]
    else:
        rows = [evaluate([m], args.subset) for m in args.model]
    if len(rows) > 1:
        t = table(rows)
        print("\n" + t)
        (RESULTS_DIR / f"comparison_{args.subset}.md").write_text(t + "\n")


if __name__ == "__main__":
    main()
