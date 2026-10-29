"""Clip listing, train/val/test splits and the PyTorch dataset.

Split modes
  random : clips split 70/15/15 at random (same people can be in train and test)
  xsub   : cross-subject - test people never appear in training (honest generalisation)
           webcam: pass --test-subjects;  ntu: official 20 training subjects
  xview  : cross-view (ntu only) - train on cameras 2 & 3, test on camera 1

Webcam clips carry the person's name in the file name: <subject>__<action>_<time>.npy
NTU clips carry it in the standard name:              S001C002P003R001A004.npy
"""
import json
import re
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from config import DATA_DIR, PROFILES, SPLIT_SEED, TEST_RATIO, VAL_RATIO
from features import fill_missing, prepare

NTU_XSUB_TRAIN = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
NTU_XVIEW_TRAIN = {2, 3}
MIN_CLIPS_PER_CLASS = 7


def clip_info(stem, profile):
    if profile == "ntu":
        m = re.search(r"C(\d{3})P(\d{3})", stem)
        return {"subject": f"P{m.group(2)}", "camera": int(m.group(1))} if m else {}
    return {"subject": stem.split("__")[0] if "__" in stem else "unknown"}


def list_clips(profile):
    root = PROFILES[profile]["keypoint_dir"]
    if not root.exists():
        raise SystemExit(f"no data in {root}")
    classes = sorted(d.name for d in root.iterdir() if d.is_dir() and any(d.glob("*.npy")))
    records = []
    for label, name in enumerate(classes):
        for f in sorted((root / name).glob("*.npy")):
            records.append({"file": f.relative_to(root).as_posix(), "label": label,
                            **clip_info(f.stem, profile)})
    return classes, records


def _stratified(records, fraction, seed=SPLIT_SEED):
    labels = [r["label"] for r in records]
    n_test = max(int(round(fraction * len(records))), len(set(labels)))   # >= 1 per class
    return train_test_split(records, test_size=n_test, stratify=labels, random_state=seed)


def split_path(profile, mode, test_subjects=None):
    name = f"split_{profile}_{mode}"
    if mode == "xsub" and profile != "ntu":
        name += "_test-" + "+".join(sorted(test_subjects))
    return DATA_DIR / f"{name}.json"


def make_split(profile, mode, test_subjects=None):
    classes, recs = list_clips(profile)
    counts = Counter(r["label"] for r in recs)
    small = [classes[c] for c in range(len(classes)) if counts[c] < MIN_CLIPS_PER_CLASS]
    if len(classes) < 2 or small:
        raise SystemExit(f"need >=2 actions and >={MIN_CLIPS_PER_CLASS} clips each; too few: {small}")

    if mode == "random":
        train, test = _stratified(recs, TEST_RATIO)
        train, val = _stratified(train, VAL_RATIO / (1 - TEST_RATIO))
    elif mode == "xsub" and profile == "ntu":
        train = [r for r in recs if int(r["subject"][1:]) in NTU_XSUB_TRAIN]
        test = [r for r in recs if int(r["subject"][1:]) not in NTU_XSUB_TRAIN]
        train, val = _stratified(train, 0.05)
    elif mode == "xview" and profile == "ntu":
        train = [r for r in recs if r["camera"] in NTU_XVIEW_TRAIN]
        test = [r for r in recs if r["camera"] not in NTU_XVIEW_TRAIN]
        train, val = _stratified(train, 0.05)
    elif mode == "xsub":
        subjects = sorted({r["subject"] for r in recs})
        if not test_subjects:
            raise SystemExit(f"cross-subject split needs --test-subjects, e.g. one of {subjects}")
        missing = set(test_subjects) - set(subjects)
        if missing:
            raise SystemExit(f"unknown subjects {missing}; recorded subjects are {subjects}")
        test = [r for r in recs if r["subject"] in test_subjects]
        rest = [r for r in recs if r["subject"] not in test_subjects]
        if not rest:
            raise SystemExit("no training subjects left - record at least two people")
        train, val = _stratified(rest, VAL_RATIO)
    else:
        raise SystemExit(f"split mode {mode!r} not available for profile {profile!r}")

    split = {"profile": profile, "mode": mode, "test_subjects": sorted(test_subjects or []),
             "classes": classes,
             "train": [[r["file"], r["label"]] for r in train],
             "val": [[r["file"], r["label"]] for r in val],
             "test": [[r["file"], r["label"]] for r in test]}
    path = split_path(profile, mode, test_subjects)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1))
    print(f"split {path.name}: {len(train)} train / {len(val)} val / {len(test)} test clips, "
          f"{len(classes)} classes")
    return path


def load_split(path):
    return json.loads(open(path).read())


class SkeletonDataset(Dataset):
    """Returns (features (C, T, V), label, clip index).

    window mode   : evaluation uses fixed windows every `stride` frames; training draws a
                    random window per item, with speed augmentation (0.8x - 1.25x).
    resample mode : one item per clip; training takes a random 50-100 % crop, both
                    resample to seq_len frames (lazy loading, suits big datasets like NTU).
    """

    def __init__(self, items, profile, feat, seq_len, train=False, speed_aug=True):
        cfg = PROFILES[profile]
        self.root, self.mode = cfg["keypoint_dir"], cfg["sample_mode"]
        self.feat, self.seq_len, self.train, self.speed_aug = feat, seq_len, train, speed_aug
        self.files = [f for f, _ in items]
        self.labels_ = [lab for _, lab in items]
        self.clips, self.index = [], []
        if self.mode == "window":
            stride = cfg["stride"]
            for ci, f in enumerate(self.files):
                seq = self._load(f)
                if len(seq) < seq_len:
                    seq = np.concatenate([seq, np.repeat(seq[-1:], seq_len - len(seq), axis=0)])
                self.clips.append(seq)
                for start in range(0, len(seq) - seq_len + 1, stride):
                    self.index.append((ci, start))
        else:
            self.index = [(ci, 0) for ci in range(len(self.files))]

    def _load(self, f):
        return fill_missing(np.load(self.root / f).astype(np.float32))

    def __len__(self):
        return len(self.index)

    def labels(self):
        return [self.labels_[ci] for ci, _ in self.index]

    def _sample(self, seq, start, rng):
        L, T = self.seq_len, len(seq)
        if self.mode == "window":
            if rng is None:
                return seq[start:start + L]
            span = int(round(L * rng.uniform(0.8, 1.25))) if self.speed_aug else L
            span = max(2, min(span, T))
            s = int(rng.integers(0, T - span + 1))
        else:
            if rng is None:
                s, span = 0, T
            else:
                span = max(1, int(round(T * rng.uniform(0.5, 1.0))))
                s = int(rng.integers(0, T - span + 1))
        idx = np.linspace(s, s + span - 1, L).round().astype(int)
        return seq[idx]

    def __getitem__(self, i):
        ci, start = self.index[i]
        seq = self.clips[ci] if self.clips else self._load(self.files[ci])
        rng = np.random.default_rng() if self.train else None
        x = prepare(self._sample(seq, start, rng), self.feat, rng)
        return torch.from_numpy(x), self.labels_[ci], ci
