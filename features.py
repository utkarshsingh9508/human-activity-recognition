"""Turns a raw skeleton window into the model input tensor (pure NumPy).

A feature config (a plain dict saved inside every checkpoint) controls what the
model sees, which is what the ablation study switches on and off:
    layout       "mediapipe" | "ntu"
    virtual      add mid-hip + neck joints (mediapipe only)
    use_z        keep the depth coordinate
    use_velocity add frame-to-frame motion (dx, dy, dz)
    stream       "joint" (positions) | "bone" (child - parent vectors, 2s-AGCN)
Training, evaluation, export and real-time all call prepare(), so they always agree.
"""
import numpy as np

from layouts import get_layout

DEFAULT_FEATURES = dict(layout="mediapipe", virtual=True, use_z=True,
                        use_velocity=True, stream="joint")


def layout_of(feat):
    return get_layout(feat["layout"], feat.get("virtual", True) and feat["layout"] == "mediapipe")


def num_channels(feat):
    d = 3 if feat["use_z"] else 2
    return d + 1 + (d if feat["use_velocity"] else 0)      # coords + visibility + velocity


def fill_missing(seq):
    """(T, J, 4) with all-zero frames where no person was found -> forward/back-filled."""
    seq = seq.copy()
    valid = seq[:, :, 3].sum(axis=1) > 0
    if not valid.any():
        return seq
    first = int(np.argmax(valid))
    seq[:first] = seq[first]
    for t in range(first + 1, len(seq)):
        if not valid[t]:
            seq[t] = seq[t - 1]
    return seq


def normalize(win, lay):
    """(T, J, 4) -> (T, V, 4): body-centred, torso-length scaled, virtual joints added.
    One centre per window keeps whole-body motion (walking, falling)."""
    win = win.astype(np.float32)
    xyz, vis = win[:, :, :3], win[:, :, 3]
    origin = xyz[:, lay.origin_joints].mean(axis=1)
    top = xyz[:, lay.top_joints].mean(axis=1)
    center = origin.mean(axis=0)
    scale = float(np.median(np.linalg.norm((top - origin)[:, :lay.torso_dims], axis=1)))
    if scale < 1e-6:
        scale = 1.0
    out = np.zeros((win.shape[0], lay.num_nodes, 4), dtype=np.float32)
    out[:, :lay.raw_joints, :3] = (xyz - center) / scale
    out[:, :lay.raw_joints, 3] = vis
    for k, joints in enumerate(lay.virtual):
        n = lay.raw_joints + k
        out[:, n, :3] = (xyz[:, joints].mean(axis=1) - center) / scale
        out[:, n, 3] = vis[:, joints].min(axis=1)
    return out


def _rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def augment(win, lay, rng):
    """Spatial 3D augmentation: mirror, yaw +-30 deg, roll +-10 deg, scale, shift, jitter."""
    win = win.copy()
    if rng.random() < 0.5:
        win = win[:, lay.flip_index()]
        win[:, :, 0] *= -1
    xyz = win[:, :, :3]
    xyz[:, :, [0, 2]] = xyz[:, :, [0, 2]] @ _rot(np.deg2rad(rng.uniform(-30, 30))).T
    xyz[:, :, :2] = xyz[:, :, :2] @ _rot(np.deg2rad(rng.uniform(-10, 10))).T
    xyz *= rng.uniform(0.9, 1.1)
    xyz[:, :, :2] += rng.uniform(-0.1, 0.1, size=2)
    xyz += rng.normal(0, 0.01, size=xyz.shape).astype(np.float32)
    return win


def to_bones(win, lay):
    """Joint positions -> bone vectors (joint minus its parent); root bone = 0."""
    parents = np.array(lay.parents)
    out = win.copy()
    out[:, :, :3] = win[:, :, :3] - win[:, parents, :3]
    out[:, :, 3] = np.minimum(win[:, :, 3], win[:, parents, 3])
    return out


def prepare(raw_win, feat, rng=None):
    """Raw (T, J, 4) window -> float32 (C, T, V). Pass rng to augment (training only)."""
    lay = layout_of(feat)
    win = normalize(raw_win, lay)
    if rng is not None:
        win = augment(win, lay, rng)
    if feat["stream"] == "bone":
        win = to_bones(win, lay)
    coords = win[:, :, :3] if feat["use_z"] else win[:, :, :2]
    parts = [coords, win[:, :, 3:4]]
    if feat["use_velocity"]:
        vel = np.zeros_like(coords)
        vel[1:] = coords[1:] - coords[:-1]
        parts.append(vel)
    x = np.concatenate(parts, axis=2)
    return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)
