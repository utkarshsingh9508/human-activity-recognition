"""Train a skeleton model.

Basic
    python train.py                                   # ST-GCN, webcam data, random split
    python train.py --model cnn_lstm                  # baseline
    python train.py --stream bone                     # bone stream (for two-stream fusion)
    python train.py --model agcn                      # adaptive graph
Cross-subject
    python train.py --split xsub --test-subjects priya
NTU RGB+D benchmark (after ntu_convert.py)
    python train.py --profile ntu --split xsub
Ablation switches
    --no-z --no-velocity --no-virtual --partition {spatial,distance,uniform}
    --no-edge-importance --no-speed-aug
"""
import argparse
import random
import time
from collections import Counter

import utils  # noqa: F401  (must come before torch)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as C
from dataset import SkeletonDataset, load_split, make_split, split_path
from evaluate import evaluate
from features import DEFAULT_FEATURES
from models import MODEL_NAMES, build_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="webcam", choices=list(C.PROFILES))
    ap.add_argument("--split", default="random", choices=["random", "xsub", "xview"])
    ap.add_argument("--test-subjects", nargs="+", default=None)
    ap.add_argument("--resplit", action="store_true", help="rebuild the split (after new data)")
    ap.add_argument("--model", default="stgcn", choices=MODEL_NAMES)
    ap.add_argument("--stream", default="joint", choices=["joint", "bone"])
    ap.add_argument("--partition", default="spatial", choices=["spatial", "distance", "uniform"])
    ap.add_argument("--no-edge-importance", action="store_true")
    ap.add_argument("--no-virtual", action="store_true")
    ap.add_argument("--no-z", action="store_true")
    ap.add_argument("--no-velocity", action="store_true")
    ap.add_argument("--no-speed-aug", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    ap.add_argument("--tag", default=None, help="name of the checkpoint/results")
    return ap.parse_args()


def default_tag(a):
    tag = a.model + ("_bone" if a.stream == "bone" else "")
    if a.split != "random":
        tag += f"_{a.split}"
    if a.profile != "webcam":
        tag = f"{a.profile}_{tag}"
    return tag


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total / n, correct / n


def plot_history(hist, path, title):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for key, ax, t in (("loss", a1, "Loss"), ("acc", a2, "Accuracy")):
        ax.plot(hist[f"train_{key}"], label="train")
        ax.plot(hist[f"val_{key}"], label="val")
        ax.set_title(t)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = utils.get_device(a.device)
    prof = C.PROFILES[a.profile]
    epochs = a.epochs or prof["epochs"]
    tag = a.tag or default_tag(a)

    sp = split_path(a.profile, a.split, a.test_subjects)
    if a.resplit or not sp.exists():
        sp = make_split(a.profile, a.split, a.test_subjects)
    split = load_split(sp)
    classes = split["classes"]

    feat = dict(DEFAULT_FEATURES, layout=prof["layout"],
                virtual=not a.no_virtual and prof["layout"] == "mediapipe",
                use_z=not a.no_z, use_velocity=not a.no_velocity, stream=a.stream)
    arch = dict(name=a.model, partition=a.partition, edge_importance=not a.no_edge_importance)
    meta = dict(tag=tag, profile=a.profile, classes=classes, seq_len=prof["seq_len"],
                features=feat, arch=arch, split_file=sp.relative_to(C.ROOT).as_posix())

    train_ds = SkeletonDataset(split["train"], a.profile, feat, prof["seq_len"], train=True,
                               speed_aug=not a.no_speed_aug)
    val_ds = SkeletonDataset(split["val"], a.profile, feat, prof["seq_len"], train=False)
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                          drop_last=len(train_ds) > a.batch, persistent_workers=a.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=a.batch, num_workers=a.workers)

    counts = Counter(train_ds.labels())
    weights = torch.tensor([len(train_ds) / (len(classes) * max(counts[i], 1))
                            for i in range(len(classes))], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    model = build_model(arch, feat, len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print(f"[{tag}] device={device}  params={sum(p.numel() for p in model.parameters()):,}  "
          f"train={len(train_ds)}  val={len(val_ds)}  classes={len(classes)}")

    C.CHECKPOINT_DIR.mkdir(exist_ok=True)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    ckpt_path = C.CHECKPOINT_DIR / f"{tag}.pt"
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best, bad = -1.0, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_dl, criterion, device, optimizer)
        vl, va = run_epoch(model, val_dl, criterion, device)
        scheduler.step()
        for k, v in zip(hist, (tl, ta, vl, va)):
            hist[k].append(v)
        mark = ""
        if va > best:
            best, bad, mark = va, 0, "  * saved"
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({**meta, "state_dict": state}, ckpt_path)
        else:
            bad += 1
        print(f"epoch {epoch:3d}  train {tl:.4f}/{ta:.4f}  val {vl:.4f}/{va:.4f}  "
              f"{time.time() - t0:.1f}s{mark}")
        if bad >= C.PATIENCE:
            print(f"early stopping after {C.PATIENCE} epochs without improvement")
            break

    plot_history(hist, C.RESULTS_DIR / f"{tag}_curves.png", tag)
    print(f"\nbest val acc {best:.4f} -> {ckpt_path}\n")
    evaluate([ckpt_path], "test", name=tag)


if __name__ == "__main__":
    main()
