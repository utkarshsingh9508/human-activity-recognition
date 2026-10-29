"""Real-time skeleton-based activity recognition.

    python realtime.py --model checkpoints/stgcn.pt                         # one model
    python realtime.py --model checkpoints/stgcn.pt checkpoints/stgcn_bone.pt  # two-stream
    python realtime.py --model exports/stgcn_int8.onnx --complexity 0       # edge mode
    python realtime.py --model exports/stgcn_int8.onnx --source clip.mp4 --save out.mp4

Press 'q' to quit. With an .onnx model this script needs no PyTorch
(see requirements-edge.txt).
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np

import config as C
from inference import Ensemble
from pose_utils import PoseExtractor, draw_skeleton

MAX_MISSING = 10      # frames without a person before the buffer is reset


def put(frame, text, org, color=(255, 255, 255), scale=0.7, thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def draw_panel(frame, classes, probs, label, conf, fps, pose_ms, model_ms, backend):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 42), (30, 30, 30), -1)
    color = (0, 220, 0) if conf >= C.CONF_THRESHOLD else (0, 200, 255)
    put(frame, f"{label}  {conf:.0%}" if conf else label, (10, 30), color, 0.9)
    if probs is not None:
        for rank, i in enumerate(np.argsort(probs)[::-1][:3]):
            y = 70 + rank * 28
            cv2.rectangle(frame, (10, y - 14), (10 + int(160 * probs[i]), y + 6), (200, 140, 40), -1)
            put(frame, f"{classes[i]} {probs[i]:.2f}", (178, y + 2), scale=0.55, thick=1)
    put(frame, f"FPS {fps:4.1f} | pose {pose_ms:4.1f} ms | model {model_ms:4.1f} ms | {backend}",
        (10, h - 12), scale=0.5, thick=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True, help=".pt or .onnx; several = fused")
    ap.add_argument("--source", default="0", help="webcam index or video path")
    ap.add_argument("--save", default=None, help="optional output video path")
    ap.add_argument("--complexity", type=int, default=1, help="MediaPipe model 0/1/2")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for the model")
    args = ap.parse_args()

    pred = Ensemble(args.model, threads=args.threads)
    if pred.features["layout"] != "mediapipe":
        raise SystemExit("real-time needs a model trained on MediaPipe (webcam) data")
    classes, seq_len = pred.classes, pred.seq_len

    is_cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source {args.source}")
    pose = PoseExtractor(model_complexity=args.complexity)
    writer = None

    buffer = deque(maxlen=seq_len)
    last_kp, missing, frame_idx = None, 0, 0
    probs_ema, model_ms, pose_ms, fps, t_prev = None, 0.0, 0.0, 0.0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if is_cam:
            frame = cv2.flip(frame, 1)                 # same mirroring as collect_data.py
        t0 = time.perf_counter()
        kp, res = pose(frame)
        pose_ms = 0.9 * pose_ms + 0.1 * (time.perf_counter() - t0) * 1000

        if kp is None:
            missing += 1
            kp = last_kp
        else:
            missing, last_kp = 0, kp
        if kp is None or missing > MAX_MISSING:
            buffer.clear()
            probs_ema = None
        else:
            buffer.append(kp)

        if len(buffer) == seq_len and frame_idx % C.PREDICT_EVERY == 0:
            t0 = time.perf_counter()
            p = pred.probs_raw(np.stack(buffer))
            model_ms = (time.perf_counter() - t0) * 1000
            probs_ema = p if probs_ema is None else C.EMA_ALPHA * p + (1 - C.EMA_ALPHA) * probs_ema

        if probs_ema is not None:
            best = int(probs_ema.argmax())
            conf = float(probs_ema[best])
            label = classes[best] if conf >= C.CONF_THRESHOLD else "uncertain"
        elif not buffer:
            label, conf = "No person", 0.0
        else:
            label, conf = f"Collecting frames {len(buffer)}/{seq_len}", 0.0

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        draw_skeleton(frame, res)
        draw_panel(frame, classes, probs_ema, label, conf, fps, pose_ms, model_ms, pred.backend)

        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                         cap.get(cv2.CAP_PROP_FPS) or 25, (w, h))
            writer.write(frame)
        cv2.imshow("Skeleton-based HAR (ST-GCN)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_idx += 1

    cap.release()
    pose.close()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
