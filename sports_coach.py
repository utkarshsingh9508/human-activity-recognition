"""AI fitness coach: real-time rep counting + form feedback.

    python sports_coach.py --exercise squat
    python sports_coach.py --exercise jumping_jack --voice          # spoken feedback (macOS)
    python sports_coach.py --exercise auto --model checkpoints/stgcn.pt
    python sports_coach.py --exercise squat --source workout.mp4 --save coached.mp4

auto  = ST-GCN recognises WHICH exercise you are doing (needs classes such as
        "squatting" / "jumping_jacks" in the model), geometry counts reps and checks form.
Keys: q quit | r reset counts | 1 squat | 2 jumping jack | a auto-detect
A session report (JSON + chart) is saved to results/sessions/ when you quit.
"""
import argparse
import json
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import config as C
from exercises import COUNTERS, SQUAT, JACK, exercise_for_class
from pose_utils import PoseExtractor, draw_skeleton, world_keypoints

GREEN, ORANGE, RED, WHITE, DARK = (80, 200, 80), (0, 165, 255), (60, 60, 230), (255, 255, 255), (35, 30, 30)
LABELS = {"squat": "SQUAT", "jumping_jack": "JUMPING JACK"}


def put(img, text, org, scale=0.7, color=WHITE, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class Voice:
    """Non-blocking text-to-speech with the macOS `say` command."""

    def __init__(self, enabled):
        self.enabled = enabled and sys.platform == "darwin"
        self.proc = None

    def say(self, text):
        if self.enabled and (self.proc is None or self.proc.poll() is not None):
            self.proc = subprocess.Popen(["say", "-r", "210", text])


def spoken(ev, counter):
    """Short phrase to speak: the rep number, or the first correction."""
    if ev.kind != "rep":
        return ev.text
    parts = ev.text.split(" - ")
    if ev.good:
        return str(counter.valid_reps)
    if parts[0].startswith("Not counted"):
        return "Not counted. " + parts[1]
    return f"{counter.valid_reps}. {parts[1]}"


def draw_panel(frame, counter, active, detector_text, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (250, 230), DARK, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    put(frame, LABELS.get(active, "choose: 1 / 2"), (12, 30), 0.7, ORANGE)
    if counter:
        put(frame, str(counter.valid_reps), (12, 115), 3.0, WHITE, 6)
        put(frame, "reps", (130, 110), 0.8, WHITE)
        s = counter.summary()
        put(frame, f"partial: {counter.partial_reps}", (12, 150), 0.6)
        put(frame, f"form score: {s['form_score'] if s['form_score'] is not None else '-'}%", (12, 178), 0.6)
        put(frame, f"phase: {counter.phase}", (12, 206), 0.6)
        # vertical range-of-motion gauge on the right edge
        x0, y0, y1 = w - 45, 60, h - 110
        cv2.rectangle(frame, (x0, y0), (x0 + 25, y1), DARK, -1)
        p = counter.progress()
        fill = int((y1 - y0) * p)
        cv2.rectangle(frame, (x0, y1 - fill), (x0 + 25, y1), GREEN if p >= 0.999 else ORANGE, -1)
        cv2.line(frame, (x0 - 6, y0), (x0 + 31, y0), WHITE, 2)
        put(frame, "target", (x0 - 75, y0 + 5), 0.5)
    if detector_text:
        put(frame, detector_text, (12, h - 70), 0.55, (200, 200, 200), 1)
    put(frame, f"FPS {fps:4.1f}", (w - 110, 25), 0.55, WHITE, 1)


def draw_message(frame, msg):
    if not msg:
        return
    text, color = msg
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x = max(10, (w - tw) // 2)
    cv2.rectangle(frame, (x - 12, h - 52), (x + tw + 12, h - 12), color, -1)
    cv2.putText(frame, text, (x, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)


def draw_joint_angle(frame, counter, kp):
    """Show the measured angle next to the joint that matters."""
    if kp is None or counter is None:
        return
    if counter.name == "squat" and hasattr(counter, "_knee"):
        j = 25 if kp[25, 3] >= kp[26, 3] else 26
        put(frame, f"{counter._knee:.0f} deg", (int(kp[j, 0]) + 15, int(kp[j, 1])), 0.7, ORANGE)
    elif counter.name == "jumping_jack" and hasattr(counter, "_arms"):
        put(frame, f"arms {counter._arms:.0f} deg", (int(kp[0, 0]) - 60, int(kp[0, 1]) - 40), 0.7, ORANGE)


def save_session(counters, started):
    out_dir = C.RESULTS_DIR / "sessions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%d_%H%M%S")
    saved = []
    for name, c in counters.items():
        if not c.trace:
            continue
        s = c.summary()
        (out_dir / f"{name}_{stamp}.json").write_text(json.dumps(s, indent=1))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t, v = np.array(c.trace).T
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(t, v, color="#16213a", lw=1.5)
        for r in c.reps:
            ax.axvspan(r["t_start"], r["t_end"], alpha=0.25,
                       color="#3cb371" if r["clean"] else ("#f2a65a" if r["valid"] else "#e05050"))
        if name == "squat":
            ax.axhline(SQUAT["depth_good"], ls="--", color="gray", lw=1, label="depth target")
            ax.invert_yaxis()
        else:
            ax.axhline(JACK["arms_up"], ls="--", color="gray", lw=1, label="arms-up target")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(c.metric_name)
        ax.set_title(f"{name}: {s['valid_reps']} reps, {s['partial_reps']} partial, "
                     f"form score {s['form_score']}%  (green = clean, orange = form issue, red = not counted)",
                     fontsize=10)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_{stamp}.png", dpi=150)
        plt.close(fig)
        saved.append(name)
        print(f"\n{name}: {s['valid_reps']} valid reps, {s['partial_reps']} partial, "
              f"form score {s['form_score']}%, issues {s['common_issues']}")
    if saved:
        print(f"session report saved in {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exercise", default="squat", choices=["squat", "jumping_jack", "auto"])
    ap.add_argument("--model", nargs="+", default=None, help="ST-GCN checkpoint(s) for --exercise auto")
    ap.add_argument("--source", default="0", help="webcam index or video file")
    ap.add_argument("--save", default=None, help="write the annotated video here")
    ap.add_argument("--complexity", type=int, default=1, help="MediaPipe model 0/1/2 (2 = most accurate)")
    ap.add_argument("--voice", action="store_true", help="speak the feedback (macOS)")
    args = ap.parse_args()

    detector = None
    if args.exercise == "auto" or args.model:
        if not args.model:
            raise SystemExit("--exercise auto needs --model checkpoints/<model>.pt")
        from inference import Ensemble
        detector = Ensemble(args.model)
        known = {c: exercise_for_class(c) for c in detector.classes}
        if not any(known.values()):
            print("warning: the model has no squat / jumping-jack class; record e.g. "
                  "'squatting' and 'jumping_jacks' and retrain for auto mode")
    active = None if args.exercise == "auto" else args.exercise
    auto = args.exercise == "auto"

    is_cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    pose = PoseExtractor(model_complexity=args.complexity)
    counters = {name: cls() for name, cls in COUNTERS.items()}
    voice = Voice(args.voice)
    buffer = deque(maxlen=detector.seq_len if detector else 1)
    candidate, streak, detector_text = None, 0, ""
    message, message_until = None, 0.0
    writer, frame_idx, fps, t_prev = None, 0, 0.0, time.time()
    started, t0 = datetime.now(), time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if is_cam:
            frame = cv2.flip(frame, 1)
        t = time.time() - t0 if is_cam else frame_idx / src_fps
        kp, res = pose(frame)
        world = world_keypoints(res)

        # --- which exercise? (ST-GCN) ---
        if detector is not None and kp is not None:
            buffer.append(kp)
            if len(buffer) == buffer.maxlen and frame_idx % C.PREDICT_EVERY == 0:
                probs = detector.probs_raw(np.stack(buffer))
                k = int(probs.argmax())
                ex = known.get(detector.classes[k])
                detector_text = f"ST-GCN: {detector.classes[k]} {probs[k]:.0%}"
                if auto:
                    streak = streak + 1 if (ex == candidate and ex and probs[k] > 0.7) else 0
                    candidate = ex
                    if ex and ex != active and streak >= 8:
                        active = ex
                        message, message_until = (f"Detected {LABELS[ex].lower()}", GREEN), t + 2.0

        # --- count reps + check form ---
        counter = counters.get(active)
        if world is not None and counter is not None:
            for ev in counter.update(world, kp, t):
                color = GREEN if ev.good else (RED if ev.text.startswith("Not counted") else ORANGE)
                message, message_until = (ev.text, color), t + 2.5
                voice.say(spoken(ev, counter))
        elif kp is None:
            message, message_until = ("Step into view - full body visible", RED), t + 0.5

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        draw_skeleton(frame, res)
        draw_joint_angle(frame, counter, kp)
        draw_panel(frame, counter, active, detector_text, fps)
        draw_message(frame, message if t < message_until else None)

        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
            writer.write(frame)
        cv2.imshow("AI Fitness Coach", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            counters = {name: cls() for name, cls in COUNTERS.items()}
        if key == ord("1"):
            active, auto = "squat", False
        if key == ord("2"):
            active, auto = "jumping_jack", False
        if key == ord("a") and detector is not None:
            auto = True
        frame_idx += 1

    cap.release()
    pose.close()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    save_session(counters, started)


if __name__ == "__main__":
    main()
