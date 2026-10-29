"""Convert a folder of videos into skeleton clips.

Expected layout (one sub-folder per action, any video format OpenCV can read):
    my_videos/
        walking/  a.mp4  b.avi ...
        running/  ...

Example:
    python extract_from_videos.py --src my_videos --fps 30
    python extract_from_videos.py --src KTH --upscale 3     # small, low-res videos

Output: data/keypoints/<action>/<video_name>.npy, shape (T, 33, 4).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import PROFILES
KEYPOINT_DIR = PROFILES["webcam"]["keypoint_dir"]
from pose_utils import PoseExtractor, empty_keypoints

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}


def extract(video_path: Path, pose: PoseExtractor, target_fps: float, upscale: float):
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    step = max(1.0, src_fps / target_fps)      # drop frames to match the webcam fps
    frames, i, next_keep = [], 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= next_keep:
            next_keep += step
            if upscale != 1:
                frame = cv2.resize(frame, None, fx=upscale, fy=upscale,
                                   interpolation=cv2.INTER_CUBIC)
            kp, _ = pose(frame)
            frames.append(kp if kp is not None else empty_keypoints())
        i += 1
    cap.release()
    return np.stack(frames) if frames else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--subject", default="video", help="person name, for cross-subject tests")
    ap.add_argument("--fps", type=float, default=30.0, help="match your webcam fps")
    ap.add_argument("--upscale", type=float, default=1.0)
    ap.add_argument("--min-detected", type=float, default=0.6)
    args = ap.parse_args()

    src = Path(args.src)
    videos = [p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXT]
    if not videos:
        raise SystemExit(f"No videos found under {src}")
    pose = PoseExtractor()
    kept = skipped = 0
    for vid in tqdm(videos):
        action = vid.relative_to(src).parts[0]
        seq = extract(vid, pose, args.fps, args.upscale)
        if seq is None or (seq[:, :, 3].sum(axis=1) > 0).mean() < args.min_detected:
            skipped += 1
            continue
        out = KEYPOINT_DIR / action / f"{args.subject}__{vid.stem}.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, seq)
        kept += 1
    pose.close()
    print(f"saved {kept} clips, skipped {skipped} (person not visible enough)")


if __name__ == "__main__":
    main()
