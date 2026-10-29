"""Record your own skeleton dataset from a webcam.

Example:
    python collect_data.py --subject utkarsh --action waving --clips 40 --seconds 4

Each clip is saved as data/keypoints/<action>/<subject>__<action>_<time>.npy,
shape (T, 33, 4). The subject name enables cross-subject testing.
Record every action with the SAME camera you will use in the live demo, from
different distances/angles, with different people and clothes if possible.
Press 'q' at any time to stop.
"""
import argparse
import time

import cv2
import numpy as np

from config import PROFILES
KEYPOINT_DIR = PROFILES["webcam"]["keypoint_dir"]
from pose_utils import PoseExtractor, draw_skeleton, empty_keypoints


def put(frame, text, y, color=(255, 255, 255), scale=0.8):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="who is recording, e.g. utkarsh")
    ap.add_argument("--action", required=True, help="class name, e.g. waving")
    ap.add_argument("--clips", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--countdown", type=float, default=2.0)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    if "__" in args.subject or "/" in args.subject:
        raise SystemExit("subject name must not contain '__' or '/'")
    out_dir = KEYPOINT_DIR / args.action
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam")
    pose = PoseExtractor()

    try:
        for clip in range(args.clips):
            # --- countdown so you can get into position ---
            t0 = time.time()
            while time.time() - t0 < args.countdown:
                ok, frame = cap.read()
                if not ok:
                    raise SystemExit("Camera read failed")
                frame = cv2.flip(frame, 1)
                _, res = pose(frame)
                draw_skeleton(frame, res)
                left = args.countdown - (time.time() - t0)
                put(frame, f"{args.subject} - {args.action}: clip {clip + 1}/{args.clips}", 30)
                put(frame, f"Get ready... {left:.1f}s", 65, (0, 200, 255))
                cv2.imshow("Collect", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return

            # --- record ---
            frames, t0 = [], time.time()
            while time.time() - t0 < args.seconds:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                kp, res = pose(frame)
                frames.append(kp if kp is not None else empty_keypoints())
                draw_skeleton(frame, res)
                put(frame, f"RECORDING {args.action} {clip + 1}/{args.clips}", 30, (0, 0, 255))
                if kp is None:
                    put(frame, "No person detected!", 65, (0, 0, 255))
                cv2.imshow("Collect", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return

            seq = np.stack(frames)
            detected = (seq[:, :, 3].sum(axis=1) > 0).mean()
            if detected < 0.7:
                print(f"clip {clip + 1}: person visible in only {detected:.0%} of frames - discarded")
                continue
            path = out_dir / f"{args.subject}__{args.action}_{int(time.time() * 1000)}.npy"
            np.save(path, seq)
            print(f"saved {path.name}  frames={len(seq)}  detected={detected:.0%}")
    finally:
        cap.release()
        pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
