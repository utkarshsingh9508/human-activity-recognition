"""Synthetic MediaPipe-style skeletons, used ONLY by selftest.py to check that the
installation and the whole pipeline work before you record real data.
(Never report results on synthetic data.)"""
import numpy as np


def _body(rng, knee=175.0, lean=8.0, arm_l=10.0, arm_r=10.0, feet=0.25, shift=0.0, scale=1.0):
    """One frame of a 33-joint world skeleton in metres (y up)."""
    w = np.zeros((33, 3))
    b = np.radians((180 - knee) / 2)
    for side, sx in ((0, -1), (1, 1)):
        ank = np.array([sx * feet / 2 + shift, 0, 0])
        kn = ank + 0.45 * np.array([0, np.cos(b), np.sin(b)])
        hip = kn + 0.45 * np.array([0, np.cos(b), -np.sin(b)])
        hip[0] = sx * 0.12 + shift
        w[[27, 28][side]], w[[25, 26][side]], w[[23, 24][side]] = ank, kn, hip
        w[[29, 30][side]] = ank + [0, -0.03, -0.05]          # heel
        w[[31, 32][side]] = ank + [0, -0.05, 0.12]           # foot index
    hipc = (w[23] + w[24]) / 2
    L = np.radians(lean)
    for side, sx, arm in ((0, -1, arm_l), (1, 1, arm_r)):
        sh = hipc + 0.5 * np.array([0, np.cos(L), np.sin(L)])
        sh[0] = sx * 0.18 + shift
        a = np.radians(arm)
        wr = sh + 0.6 * np.array([sx * np.sin(a), -np.cos(a), 0])
        w[[11, 12][side]], w[[15, 16][side]], w[[13, 14][side]] = sh, wr, (sh + wr) / 2
        for k, off in zip(([17, 19, 21], [18, 20, 22])[side], ([0.05, 0], [0.07, 0.01], [0.04, 0.03])):
            w[k] = wr + [sx * off[0], -0.05, off[1]]
    neck = (w[11] + w[12]) / 2
    w[0] = neck + [0, 0.22, 0.05]
    for k, off in zip(range(1, 11), ([-.02, .03], [-.04, .03], [-.06, .03], [.02, .03], [.04, .03],
                                     [.06, .03], [-.08, 0], [.08, 0], [-.03, -.04], [.03, -.04])):
        w[k] = w[0] + [off[0], off[1], 0]
    return w * scale


def make_clip(action, rng, frames=60, subject_scale=1.0, speed=1.0):
    """(frames, 33, 4) in pixels like collect_data.py would save, for 640x480 video."""
    t = np.arange(frames) / 30.0 * speed
    out = np.zeros((frames, 33, 4), dtype=np.float32)
    phase = rng.uniform(0, 2 * np.pi)
    for i, ti in enumerate(t):
        s = np.sin(2 * np.pi * 0.5 * ti + phase)
        kw = {}
        if action == "squatting":
            kw = dict(knee=175 - 85 * max(0.0, s), lean=10 + 25 * max(0.0, s))
        elif action == "waving":
            kw = dict(arm_r=150 + 25 * np.sin(2 * np.pi * 1.5 * ti + phase))
        elif action == "jumping_jacks":
            p = max(0.0, np.sin(2 * np.pi * 0.8 * ti + phase))
            kw = dict(arm_l=10 + 160 * p, arm_r=10 + 160 * p, feet=0.25 + 0.5 * p)
        elif action == "walking":
            kw = dict(shift=0.4 * np.sin(2 * np.pi * 0.2 * ti + phase),
                      knee=165 + 10 * np.sin(2 * np.pi * 1.0 * ti))
        w = _body(rng, scale=subject_scale, **kw)
        w += rng.normal(0, 0.008, w.shape)
        out[i, :, 0] = 320 + w[:, 0] * 260
        out[i, :, 1] = 420 - w[:, 1] * 260
        out[i, :, 2] = -w[:, 2] * 260
        out[i, :, 3] = rng.uniform(0.85, 1.0, 33)
    return out


def world_frame(rng, **kw):
    """World landmarks (33, 4) as MediaPipe gives them (y down), for exercise tests."""
    w = _body(rng, **kw)
    w[:, 1] *= -1
    img = np.zeros((33, 4))
    img[:, 0] = 320 + w[:, 0] * 260
    img[:, 1] = 420 + w[:, 1] * 260
    out = np.concatenate([w, np.ones((33, 1))], axis=1).astype(np.float32)
    img[:, 3] = 1
    return out, img.astype(np.float32)
