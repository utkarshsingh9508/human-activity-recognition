"""Rep counting and form feedback for squats and jumping jacks (pure NumPy).

Works on MediaPipe *world* landmarks: 33 joints in metres, centred on the hips, so
joint angles do not depend on how far the person stands from the camera. All checks
are orientation-agnostic (they use angles and distances, not "up/down" in pixels).

Every exercise is a small state machine with hysteresis:
    squat         UP --(knee < 140 deg)--> DOWN --(knee > 160 deg)--> UP  = one rep
    jumping jack  CLOSED --(arms/legs start opening)--> OPEN --(arms down + feet together)--> CLOSED

During a rep the extreme values are tracked (deepest knee angle, largest torso lean, ...)
and judged when the rep finishes. Tune the numbers in SQUAT / JACK if needed.
"""
from dataclasses import dataclass, field

import numpy as np

# MediaPipe landmark indices
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 23, 24, 25, 26, 27, 28

SQUAT = dict(
    descent=140,        # knee angle below this -> a rep has started
    stand=160,          # knee angle above this -> back to standing, rep complete
    depth_good=100,     # deepest knee angle must reach this (thighs ~parallel)
    max_lean=50,        # torso angle from vertical, degrees
    valgus_min=0.75,    # knee width / ankle width at the bottom (front view only)
    min_rep_time=1.0,   # seconds; faster = "control the descent"
)
JACK = dict(
    arms_start=100,     # arm angle (hip-shoulder-wrist) where the opening begins
    arms_up=145,        # both arms must reach this: hands overhead
    arms_down=45,       # arms back by the sides
    legs_start=1.7,     # ankle distance / hip width where the jump-out begins
    legs_open=2.0,      # feet must reach this spread
    legs_closed=1.5,    # feet back together
    max_sync_gap=0.35,  # seconds between arms and legs starting to open
)
VIS_MIN = 0.5           # landmark visibility needed to trust a joint
CUE_COOLDOWN = 2.5      # seconds before the same live cue is repeated


def angle(a, b, c):
    """Angle at b (degrees) formed by points a-b-c."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def tilt_from_vertical(v):
    """Angle between vector v and the vertical axis (works for y-up or y-down)."""
    return float(np.degrees(np.arccos(np.clip(abs(v[1]) / (np.linalg.norm(v) + 1e-9), 0, 1))))


def front_view_ratio(img):
    """Shoulder width / torso length in the image: ~0.6+ facing the camera, ~0.1 side-on."""
    sh = abs(img[L_SH, 0] - img[R_SH, 0])
    torso = np.linalg.norm((img[L_SH, :2] + img[R_SH, :2]) / 2 - (img[L_HIP, :2] + img[R_HIP, :2]) / 2)
    return float(sh / (torso + 1e-9))


@dataclass
class Event:
    kind: str           # "cue" (live tip), "rep" (rep finished), "info"
    text: str
    good: bool = True


class RepCounter:
    name = "exercise"
    metric_name = "metric"

    def __init__(self, smoothing=0.5):
        self.alpha = smoothing
        self.valid_reps = 0         # reps that met the range-of-motion requirement
        self.partial_reps = 0
        self.reps = []              # per-rep details
        self.trace = []             # (t, primary metric) for the session chart
        self.phase = "ready"
        self._ema = {}
        self._last_cue = {}
        self._rep = None

    def _smooth(self, key, value):
        prev = self._ema.get(key)
        self._ema[key] = value if prev is None else self.alpha * value + (1 - self.alpha) * prev
        return self._ema[key]

    def _cue(self, key, text, t, events):
        if t - self._last_cue.get(key, -1e9) >= CUE_COOLDOWN:
            self._last_cue[key] = t
            events.append(Event("cue", text, good=False))

    def _finish(self, rep, valid, issues, events):
        rep.update(valid=valid, clean=valid and not issues, issues=issues)
        self.reps.append(rep)
        if valid:
            self.valid_reps += 1
        else:
            self.partial_reps += 1
        if valid and not issues:
            events.append(Event("rep", f"Rep {self.valid_reps}: good rep!", True))
        elif valid:
            events.append(Event("rep", f"Rep {self.valid_reps} counted - {issues[0]}", False))
        else:
            events.append(Event("rep", f"Not counted - {issues[0]}", False))

    def progress(self):
        """0..1 for the on-screen gauge."""
        return 0.0

    def summary(self):
        n = len(self.reps)
        issues = {}
        for r in self.reps:
            for i in r["issues"]:
                key = i.split(" - ")[0]
                issues[key] = issues.get(key, 0) + 1
        durations = [r["duration"] for r in self.reps]
        return {"exercise": self.name, "valid_reps": self.valid_reps,
                "partial_reps": self.partial_reps,
                "clean_reps": sum(r["clean"] for r in self.reps),
                "form_score": round(100 * sum(r["clean"] for r in self.reps) / n) if n else None,
                "avg_rep_time_s": round(float(np.mean(durations)), 2) if n else None,
                "common_issues": dict(sorted(issues.items(), key=lambda kv: -kv[1])),
                "reps": self.reps}


class SquatCounter(RepCounter):
    name = "squat"
    metric_name = "knee angle (deg)"

    def metrics(self, w, img):
        vis = w[:, 3]
        sides = []
        for hip, knee, ank, sh in ((L_HIP, L_KNEE, L_ANK, L_SH), (R_HIP, R_KNEE, R_ANK, R_SH)):
            v = min(vis[hip], vis[knee], vis[ank])
            sides.append((v, angle(w[hip, :3], w[knee, :3], w[ank, :3])))
        seen = [s for s in sides if s[0] >= VIS_MIN] or [max(sides)]
        torso = (w[L_SH, :3] + w[R_SH, :3]) / 2 - (w[L_HIP, :3] + w[R_HIP, :3]) / 2
        knee_w = abs(w[L_KNEE, 0] - w[R_KNEE, 0])
        ank_w = abs(w[L_ANK, 0] - w[R_ANK, 0])
        return dict(visible=max(s[0] for s in sides) >= 0.3,
                    knee=float(np.mean([s[1] for s in seen])),
                    lean=tilt_from_vertical(torso),
                    valgus=knee_w / (ank_w + 1e-6),
                    frontal=front_view_ratio(img) > 0.45)

    def update(self, w, img, t):
        m = self.metrics(w, img)
        events = []
        if not m["visible"]:
            return events
        knee = self._smooth("knee", m["knee"])
        lean = self._smooth("lean", m["lean"])
        self.trace.append((t, knee))
        self._knee = knee
        c = SQUAT
        if self.phase in ("ready", "up"):
            self.phase = "up"
            if knee < c["descent"]:
                self.phase = "down"
                self._rep = dict(t_start=t, min_knee=knee, max_lean=lean, min_valgus=None)
        else:
            r = self._rep
            r["min_knee"] = min(r["min_knee"], knee)
            r["max_lean"] = max(r["max_lean"], lean)
            if m["frontal"] and knee < 120:                  # knee cave only visible from the front
                r["min_valgus"] = m["valgus"] if r["min_valgus"] is None else min(r["min_valgus"], m["valgus"])
            if lean > c["max_lean"]:
                self._cue("lean", "Keep your chest up", t, events)
            if m["frontal"] and knee < 120 and m["valgus"] < c["valgus_min"]:
                self._cue("valgus", "Push your knees out", t, events)
            if knee > c["stand"]:
                self.phase = "up"
                r.update(t_end=t, duration=round(t - r["t_start"], 2),
                         min_knee=round(r["min_knee"], 1), max_lean=round(r["max_lean"], 1))
                issues = []
                valid = r["min_knee"] <= c["depth_good"]
                if not valid:
                    issues.append(f"Go lower - knees only reached {r['min_knee']:.0f} deg, "
                                  f"aim for {c['depth_good']} or less")
                if r["max_lean"] > c["max_lean"]:
                    issues.append(f"Keep your chest up - torso leaned {r['max_lean']:.0f} deg")
                if r["min_valgus"] is not None and r["min_valgus"] < c["valgus_min"]:
                    issues.append("Push your knees out - they caved inward")
                if r["duration"] < c["min_rep_time"]:
                    issues.append("Slow down - control the descent")
                self._finish(r, valid, issues, events)
                self._rep = None
        return events

    def progress(self):
        k = getattr(self, "_knee", 180.0)
        return float(np.clip((175 - k) / (175 - SQUAT["depth_good"]), 0, 1))


class JumpingJackCounter(RepCounter):
    name = "jumping_jack"
    metric_name = "arm angle (deg)"

    def metrics(self, w, img):
        vis = w[:, 3]
        arms = [angle(w[L_HIP, :3], w[L_SH, :3], w[L_WR, :3]),
                angle(w[R_HIP, :3], w[R_SH, :3], w[R_WR, :3])]
        hip_w = np.linalg.norm(w[L_HIP, [0, 2]] - w[R_HIP, [0, 2]])
        legs = abs(w[L_ANK, 0] - w[R_ANK, 0]) / (hip_w + 1e-6)
        key = [L_SH, R_SH, L_WR, R_WR, L_ANK, R_ANK]
        return dict(visible=min(vis[key]) >= 0.3, arms=min(arms), legs=float(legs))

    def update(self, w, img, t):
        m = self.metrics(w, img)
        events = []
        if not m["visible"]:
            return events
        arms = self._smooth("arms", m["arms"])
        legs = self._smooth("legs", m["legs"])
        self.trace.append((t, arms))
        self._arms, self._legs = arms, legs
        c = JACK
        if self.phase in ("ready", "closed"):
            self.phase = "closed"
            if arms > c["arms_start"] or legs > c["legs_start"]:
                self.phase = "open"
                self._rep = dict(t_start=t, max_arms=arms, max_legs=legs,
                                 t_arms=t if arms > c["arms_start"] else None,
                                 t_legs=t if legs > c["legs_start"] else None)
        else:
            r = self._rep
            r["max_arms"] = max(r["max_arms"], arms)
            r["max_legs"] = max(r["max_legs"], legs)
            if r["t_arms"] is None and arms > c["arms_start"]:
                r["t_arms"] = t
            if r["t_legs"] is None and legs > c["legs_start"]:
                r["t_legs"] = t
            if (arms < c["arms_down"] and legs > c["legs_closed"]
                    and r["max_arms"] >= c["arms_start"] and r["max_legs"] >= c["legs_start"]):
                self._cue("feet", "Bring your feet back together", t, events)
            if arms < c["arms_down"] and legs < c["legs_closed"]:
                self.phase = "closed"
                r.update(t_end=t, duration=round(t - r["t_start"], 2),
                         max_arms=round(r["max_arms"], 1), max_legs=round(r["max_legs"], 2))
                arms_ok = r["max_arms"] >= c["arms_up"]
                legs_ok = r["max_legs"] >= c["legs_open"]
                issues = []
                if not arms_ok:
                    issues.append(f"Raise your arms fully overhead - reached {r['max_arms']:.0f} deg")
                if not legs_ok:
                    issues.append("Jump your feet wider")
                if arms_ok and legs_ok and r["t_arms"] is not None and r["t_legs"] is not None \
                        and abs(r["t_arms"] - r["t_legs"]) > c["max_sync_gap"]:
                    issues.append("Move arms and legs together")
                valid = arms_ok and legs_ok
                if r["duration"] < 0.25 and not valid:
                    self._rep = None          # a twitch, not a rep
                    return events
                self._finish(r, valid, issues, events)
                self._rep = None
        return events

    def progress(self):
        a = getattr(self, "_arms", 0.0)
        return float(np.clip(a / JACK["arms_up"], 0, 1))


COUNTERS = {"squat": SquatCounter, "jumping_jack": JumpingJackCounter}


def exercise_for_class(class_name):
    """Map an ST-GCN class name to a counter, e.g. 'squatting' -> 'squat'."""
    n = class_name.lower()
    if "squat" in n:
        return "squat"
    if "jack" in n or "jumping" in n:
        return "jumping_jack"
    return None
