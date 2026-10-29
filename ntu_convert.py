"""Convert NTU RGB+D 60 skeleton files into this project's format.

1. Request the dataset at https://rose1.ntu.edu.sg/dataset/actionRecognition/ and
   download "nturgbd_skeletons_s001_to_s017.zip" (3D skeletons only, ~5.8 GB).
2. Unzip it, then:
       python ntu_convert.py --src path/to/nturgb+d_skeletons --workers 8
3. Train / evaluate with the official protocols:
       python train.py --profile ntu --split xsub
       python train.py --profile ntu --split xview

Output: data/ntu/keypoints/A0xx_<action>/<sample>.npy, shape (T, 25, 4) = x, y, z
(metres, camera space) + tracking confidence (tracked 1.0, inferred 0.5, missing 0).
For two-person actions (A050-A060) the most active person is kept - the original
ST-GCN models both people, so expect slightly lower accuracy on those classes.
"""
import argparse
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import PROFILES

NTU60_ACTIONS = [
    "drink_water", "eat_meal", "brush_teeth", "brush_hair", "drop", "pick_up", "throw",
    "sit_down", "stand_up", "clapping", "reading", "writing", "tear_up_paper",
    "put_on_jacket", "take_off_jacket", "put_on_shoe", "take_off_shoe", "put_on_glasses",
    "take_off_glasses", "put_on_hat", "take_off_hat", "cheer_up", "hand_waving",
    "kicking_something", "reach_into_pocket", "hopping", "jump_up", "phone_call",
    "play_with_phone", "typing", "pointing", "taking_selfie", "check_time",
    "rub_hands", "nod_head", "shake_head", "wipe_face", "salute", "put_palms_together",
    "cross_hands", "sneeze_cough", "staggering", "falling", "headache", "chest_pain",
    "back_pain", "neck_pain", "nausea", "fan_self", "punch_other", "kick_other",
    "push_other", "pat_on_back", "point_at_other", "hug", "give_object", "touch_pocket",
    "handshake", "walk_towards", "walk_apart",
]


def read_skeleton(path):
    """Parse one .skeleton file -> (T, 25, 4) for the most active body, or None."""
    lines = Path(path).read_text().split("\n")
    pos = 0

    def nxt():
        nonlocal pos
        pos += 1
        return lines[pos - 1]

    n_frames = int(nxt())
    bodies = {}
    for t in range(n_frames):
        n_bodies = int(nxt())
        for _ in range(n_bodies):
            body_id = nxt().split()[0]
            n_joints = int(nxt())
            j = np.array([nxt().split() for _ in range(n_joints)], dtype=np.float32)
            arr = np.zeros((25, 4), dtype=np.float32)
            arr[:, :3] = j[:25, :3]
            arr[:, 3] = j[:25, 11] / 2.0
            bodies.setdefault(body_id, {})[t] = arr
    if not bodies:
        return None

    def activity(frames):
        xyz = np.stack([a[:, :3] for a in frames.values()])
        return float(xyz.std(axis=0).sum()) * len(frames)

    best = max(bodies.values(), key=activity)
    seq = np.zeros((n_frames, 25, 4), dtype=np.float32)
    for t, a in best.items():
        seq[t] = a
    valid = np.where(seq[:, :, 3].sum(axis=1) > 0)[0]
    if len(valid) < 2:
        return None
    return seq[valid[0]:valid[-1] + 1]


def convert(args):
    path, out_root, max_action = args
    m = re.search(r"A(\d{3})", path.stem)
    if not m or int(m.group(1)) > max_action:
        return "skipped"
    a = int(m.group(1))
    out = out_root / f"A{a:03d}_{NTU60_ACTIONS[a - 1]}" / f"{path.stem}.npy"
    if out.exists():
        return "exists"
    try:
        seq = read_skeleton(path)
    except (ValueError, IndexError):
        return "broken"
    if seq is None:
        return "empty"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, seq)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with *.skeleton files")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--single-person-only", action="store_true",
                    help="keep actions 1-49 only (drop two-person interactions)")
    args = ap.parse_args()
    files = sorted(Path(args.src).rglob("*.skeleton"))
    if not files:
        raise SystemExit(f"no .skeleton files under {args.src}")
    out_root = PROFILES["ntu"]["keypoint_dir"]
    max_action = 49 if args.single_person_only else 60
    jobs = [(f, out_root, max_action) for f in files]
    stats = {}
    with Pool(args.workers) as pool:
        for s in tqdm(pool.imap_unordered(convert, jobs, chunksize=32), total=len(jobs)):
            stats[s] = stats.get(s, 0) + 1
    print(stats)


if __name__ == "__main__":
    main()
