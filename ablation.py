"""Ablation study: retrain ST-GCN with one component removed at a time and
collect the test results in one table (results/ablation_<profile>_<split>.md).

    python ablation.py                                     # random split, 1 seed
    python ablation.py --split xsub --test-subjects priya  # cross-subject (recommended)
    python ablation.py --seeds 3                           # mean +- std over 3 seeds
    python ablation.py --only full no_z bone two_stream    # a subset of rows
    python ablation.py --epochs 40                         # faster, fewer epochs

Finished runs are reused, so you can stop (Ctrl+C) and continue later.
"""
import argparse
import json
import subprocess
import sys

import numpy as np

import config as C

VARIANTS = [  # key, table label, extra train.py flags
    ("full", "ST-GCN (full model)", []),
    ("no_z", "- depth (z) coordinate", ["--no-z"]),
    ("no_velocity", "- velocity features", ["--no-velocity"]),
    ("no_virtual", "- virtual joints (mid-hip, neck)", ["--no-virtual"]),
    ("distance", "distance partitioning (K=2)", ["--partition", "distance"]),
    ("uniform", "uniform partitioning (K=1)", ["--partition", "uniform"]),
    ("no_edge_imp", "- learnable edge importance", ["--no-edge-importance"]),
    ("no_speed", "- speed augmentation", ["--no-speed-aug"]),
    ("bone", "bone stream only", ["--stream", "bone"]),
    ("two_stream", "two-stream fusion (joint + bone)", None),     # built from full + bone
    ("agcn", "adaptive graph (AGCN)", ["--model", "agcn"]),
    ("cnn_lstm", "CNN-LSTM baseline (no graph)", ["--model", "cnn_lstm"]),
]


def run(tag, flags, common):
    out = C.RESULTS_DIR / f"{tag}_test.json"
    if out.exists():
        print(f"[skip] {tag} already done")
    else:
        cmd = [sys.executable, "train.py", "--tag", tag, *common, *flags]
        print("\n$ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    return json.loads(out.read_text())


def fuse(tag, joint_tag, bone_tag):
    out = C.RESULTS_DIR / f"{tag}_test.json"
    if not out.exists():
        from evaluate import evaluate
        evaluate([C.CHECKPOINT_DIR / f"{joint_tag}.pt", C.CHECKPOINT_DIR / f"{bone_tag}.pt"],
                 "test", name=tag, verbose=False)
    return json.loads(out.read_text())


def fmt(vals):
    vals = np.array(vals) * 100
    return f"{vals.mean():.2f}" if len(vals) == 1 else f"{vals.mean():.2f} ± {vals.std():.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="webcam", choices=list(C.PROFILES))
    ap.add_argument("--split", default="random")
    ap.add_argument("--test-subjects", nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--only", nargs="+", default=None, help="variant keys to run")
    args = ap.parse_args()

    common = ["--profile", args.profile, "--split", args.split]
    if args.test_subjects:
        common += ["--test-subjects", *args.test_subjects]
    if args.epochs:
        common += ["--epochs", str(args.epochs)]
    wanted = [v for v in VARIANTS if not args.only or v[0] in args.only]
    if any(v[0] == "two_stream" for v in wanted):          # fusion needs both streams
        keys = {v[0] for v in wanted} | {"full", "bone"}
        wanted = [v for v in VARIANTS if v[0] in keys]
    prefix = f"abl_{args.profile}_{args.split}" + (
        "_" + "+".join(args.test_subjects) if args.test_subjects else "")

    results = {}
    for key, label, flags in wanted:
        for s in range(args.seeds):
            tag = f"{prefix}_{key}_s{s}"
            if flags is None:
                r = fuse(tag, f"{prefix}_full_s{s}", f"{prefix}_bone_s{s}")
            else:
                r = run(tag, flags + ["--seed", str(s)], common)
            results.setdefault(key, []).append(r)

    full_clip = np.mean([r["clip_acc"] for r in results["full"]]) if "full" in results else None
    lines = [f"Ablation - profile `{args.profile}`, split `{args.split}`"
             + (f" (test: {', '.join(args.test_subjects)})" if args.test_subjects else "")
             + f", {args.seeds} seed(s). Accuracies in %.", "",
             "| Variant | Window acc | Clip acc | Δ clip vs full | Macro-F1 | CPU ms |",
             "|---|---|---|---|---|---|"]
    for key, label, _ in wanted:
        rs = results[key]
        clip = np.mean([r["clip_acc"] for r in rs])
        delta = "–" if full_clip is None or key == "full" else f"{(clip - full_clip) * 100:+.2f}"
        lines.append(f"| {label} | {fmt([r['window_acc'] for r in rs])} | "
                     f"{fmt([r['clip_acc'] for r in rs])} | {delta} | "
                     f"{np.mean([r['macro_f1'] for r in rs]):.4f} | "
                     f"{np.mean([r['latency_ms'] for r in rs]):.2f} |")
    text = "\n".join(lines)
    out = C.RESULTS_DIR / f"ablation_{prefix[4:]}.md"
    out.write_text(text + "\n")
    print("\n" + text + f"\n\nsaved {out}")


if __name__ == "__main__":
    main()
