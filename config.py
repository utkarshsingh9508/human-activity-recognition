"""Central settings. Change values here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
EXPORT_DIR = ROOT / "exports"
RESULTS_DIR = ROOT / "results"

# A profile = one dataset + how its clips are turned into training samples.
#   window   : many overlapping fixed-length windows per clip (matches real-time use)
#   resample : one sample per clip, whole clip resampled to seq_len frames (NTU protocol)
PROFILES = {
    "webcam": dict(keypoint_dir=DATA_DIR / "keypoints", layout="mediapipe",
                   seq_len=30, stride=5, sample_mode="window", epochs=80),
    "selftest": dict(keypoint_dir=DATA_DIR / "selftest" / "keypoints", layout="mediapipe",
                     seq_len=30, stride=5, sample_mode="window", epochs=3),
    "ntu": dict(keypoint_dir=DATA_DIR / "ntu" / "keypoints", layout="ntu",
                seq_len=64, stride=None, sample_mode="resample", epochs=65),
}

# Default actions for webcam recording (training reads classes from the folders)
ACTIONS = ["standing", "walking", "sitting", "waving", "clapping", "squatting", "jumping_jacks", "idle"]

# Split ratios for the "random" split
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 42            # fixed, so every experiment uses the same split

# Training defaults
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15              # early stopping on validation accuracy

# Real-time inference
PREDICT_EVERY = 2
EMA_ALPHA = 0.4
CONF_THRESHOLD = 0.6
