"""Explainability: draw the skeleton with every bone coloured by how strongly the
trained ST-GCN / AGCN uses it. For AGCN, the strongest learned *non-bone* links
(e.g. hand-to-hand) are drawn as dashed lines.

    python visualize_graph.py --ckpt checkpoints/stgcn.pt
    python visualize_graph.py --ckpt checkpoints/agcn.pt
"""
import argparse

import utils  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import PROFILES, RESULTS_DIR
from features import fill_missing, layout_of, normalize
from inference import load_checkpoint


def mean_pose(meta, lay):
    root = PROFILES[meta["profile"]]["keypoint_dir"]
    clip = next(root.rglob("*.npy"), None)
    if clip is None:
        raise SystemExit(f"no clips in {root}")
    pos = normalize(fill_missing(np.load(clip)), lay)[:, :, :2].mean(axis=0)
    if lay.y_down:
        pos[:, 1] *= -1
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--top-links", type=int, default=6)
    args = ap.parse_args()
    model, meta = load_checkpoint(args.ckpt, "cpu")
    if not hasattr(model, "blocks"):
        raise SystemExit("only graph models (stgcn / stgcn_lite / agcn) can be visualised")
    lay = layout_of(meta["features"])

    with torch.no_grad():
        mats = []
        for i, block in enumerate(model.blocks):
            A = model.A * model.edge_importance[i] if model.edge_importance is not None else model.A
            if block.gcn.adaptive:
                A = A + block.gcn.B
            mats.append(A.abs().sum(0))
        eff = torch.stack(mats).mean(0).numpy()
    sym = eff + eff.T
    np.fill_diagonal(sym, 0)

    bone_set = {tuple(sorted(e)) for e in lay.edges}
    bw = np.array([sym[i, j] for i, j in lay.edges])
    bw = (bw - bw.min()) / (np.ptp(bw) + 1e-9)
    pos = mean_pose(meta, lay)
    fig, ax = plt.subplots(figsize=(6, 8))
    cmap = plt.get_cmap("inferno_r")
    for (i, j), w in zip(lay.edges, bw):
        ax.plot(pos[[i, j], 0], pos[[i, j], 1], color=cmap(0.15 + 0.85 * w),
                linewidth=1 + 7 * w, solid_capstyle="round")

    extra = []
    if any(b.gcn.adaptive for b in model.blocks):
        iu = np.triu_indices(lay.num_nodes, 1)
        cand = [(sym[i, j], i, j) for i, j in zip(*iu) if (i, j) not in bone_set]
        extra = sorted(cand, reverse=True)[:args.top_links]
        for _, i, j in extra:
            ax.plot(pos[[i, j], 0], pos[[i, j], 1], "--", color="#1f8a78", linewidth=2)

    ax.scatter(pos[:, 0], pos[:, 1], s=12, color="#16213a", zorder=3)
    top = np.argsort(bw)[::-1][:3]
    ax.set_title(f"{meta['tag']}: learned edge importance\nstrongest: " + ", ".join(
        f"{lay.names[lay.edges[k][0]]}-{lay.names[lay.edges[k][1]]}" for k in top), fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{meta['tag']}_edge_importance.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    for _, i, j in extra:
        print(f"  learned non-bone link: {lay.names[i]} - {lay.names[j]}")


if __name__ == "__main__":
    main()
