"""Add a subject name to clips recorded without one (e.g. with the old
collect_data.py), so they can be used for cross-subject testing.

    python set_subject.py --subject utkarsh                 # all untagged clips
    python set_subject.py --subject priya --action waving   # only one action
    python set_subject.py --list                            # show clips per subject
"""
import argparse
from collections import Counter

from config import PROFILES

ROOT = PROFILES["webcam"]["keypoint_dir"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject")
    ap.add_argument("--action", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    files = sorted(ROOT.glob(f"{args.action or '*'}/*.npy"))
    if args.list or not args.subject:
        table = Counter((f.parent.name, f.stem.split("__")[0] if "__" in f.stem else "unknown")
                        for f in files)
        for (action, subj), n in sorted(table.items()):
            print(f"{action:<15} {subj:<15} {n}")
        return
    renamed = 0
    for f in files:
        if "__" not in f.stem:
            f.rename(f.with_name(f"{args.subject}__{f.name}"))
            renamed += 1
    print(f"tagged {renamed} clips with subject '{args.subject}'")


if __name__ == "__main__":
    main()
