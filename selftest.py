"""One-command check that everything works on your computer (about 2-5 minutes).

    python selftest.py

It creates a small SYNTHETIC dataset in data/selftest/ (it never touches your real
recordings), then runs every part of the project: training (ST-GCN, bone stream, AGCN,
CNN-LSTM), random and cross-subject splits, evaluation and two-stream fusion,
ONNX + INT8 export, graph visualisation, and the squat / jumping-jack counters.
Finishes with "ALL CHECKS PASSED", or tells you which step failed - send that to us.
Add --keep to keep the self-test files afterwards.
"""
import argparse
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

import config as C
import synth

ACTIONS = ["standing", "squatting", "waving", "jumping_jacks"]
SUBJECTS = {"alice": (1.0, 1.0), "bob": (1.12, 0.85)}      # body scale, speed


def step(name, fn, results):
    t0 = time.time()
    print(f"\n=== {name} ===", flush=True)
    try:
        fn()
        results.append((name, True, time.time() - t0))
        print(f"--- OK ({time.time() - t0:.0f}s)")
    except Exception:
        traceback.print_exc()
        results.append((name, False, time.time() - t0))
        print(f"--- FAILED: {name}")


def run(*args):
    subprocess.run([sys.executable, *args], check=True)


def make_data():
    root = C.PROFILES["selftest"]["keypoint_dir"]
    shutil.rmtree(root.parent, ignore_errors=True)
    rng = np.random.default_rng(0)
    for action in ACTIONS:
        (root / action).mkdir(parents=True, exist_ok=True)
        for subj, (scale, speed) in SUBJECTS.items():
            for i in range(10):
                clip = synth.make_clip(action, rng, frames=60, subject_scale=scale, speed=speed)
                np.save(root / action / f"{subj}__{action}_{i}.npy", clip)
    print(f"created {len(ACTIONS) * len(SUBJECTS) * 10} synthetic clips in {root}")


def check_exercises():
    from exercises import JumpingJackCounter, SquatCounter
    rng = np.random.default_rng(1)
    sq, t = SquatCounter(), 0.0
    for depth in (85, 90, 125):                     # 2 good reps + 1 too shallow
        for k in list(np.linspace(175, depth, 30)) + list(np.linspace(depth, 175, 30)) + [175] * 10:
            sq.update(*synth.world_frame(rng, knee=k), t)
            t += 1 / 30
    assert (sq.valid_reps, sq.partial_reps) == (2, 1), (sq.valid_reps, sq.partial_reps)
    jj, t = JumpingJackCounter(), 0.0
    for _ in range(3):
        for p in list(np.sin(np.linspace(0, np.pi, 30))) + [0] * 8:
            jj.update(*synth.world_frame(rng, arm_l=10 + 160 * p, arm_r=10 + 160 * p,
                                         feet=0.25 + 0.5 * p), t)
            t += 1 / 30
    assert jj.valid_reps == 3, jj.valid_reps
    print(f"squat: {sq.valid_reps} valid + {sq.partial_reps} partial (expected 2 + 1); "
          f"jumping jacks: {jj.valid_reps} (expected 3)")


def check_imports():
    import cv2, mediapipe, onnxruntime, sklearn, torch  # noqa: F401,E401
    import utils
    print(f"torch {torch.__version__}, device for training: {utils.get_device()}, "
          f"onnxruntime {onnxruntime.__version__}, opencv {cv2.__version__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    ep = ["--profile", "selftest", "--epochs", "3"]
    ck = C.CHECKPOINT_DIR
    results = []
    step("1. libraries and device", check_imports, results)
    step("2. synthetic dataset", make_data, results)
    step("3. train ST-GCN (random split)", lambda: run("train.py", *ep, "--tag", "selftest_stgcn"), results)
    step("4. train bone stream", lambda: run("train.py", *ep, "--stream", "bone", "--tag", "selftest_bone"), results)
    step("5. train adaptive graph (AGCN)", lambda: run("train.py", *ep, "--model", "agcn", "--tag", "selftest_agcn"), results)
    step("6. train CNN-LSTM baseline", lambda: run("train.py", *ep, "--model", "cnn_lstm", "--tag", "selftest_cnnlstm"), results)
    step("7. ablation switches", lambda: run("train.py", *ep, "--no-z", "--no-velocity", "--no-virtual",
                                             "--partition", "uniform", "--no-edge-importance",
                                             "--no-speed-aug", "--tag", "selftest_ablation"), results)
    step("8. cross-subject split", lambda: run("train.py", *ep, "--split", "xsub", "--test-subjects", "bob",
                                               "--tag", "selftest_xsub"), results)
    step("9. evaluate + two-stream fusion", lambda: run("evaluate.py", "--model", str(ck / "selftest_stgcn.pt"),
                                                        str(ck / "selftest_bone.pt"), "--fuse"), results)
    step("10. ONNX + INT8 export", lambda: run("export_edge.py", "--ckpt", str(ck / "selftest_stgcn.pt"),
                                               "--calib", "50"), results)
    step("11. graph visualisation", lambda: run("visualize_graph.py", "--ckpt", str(ck / "selftest_agcn.pt")), results)
    step("12. exercise counters", check_exercises, results)

    if not args.keep:
        shutil.rmtree(C.PROFILES["selftest"]["keypoint_dir"].parent, ignore_errors=True)
        for d in (C.CHECKPOINT_DIR, C.EXPORT_DIR, C.RESULTS_DIR):
            for f in d.glob("*selftest*"):
                f.unlink()
        for f in C.DATA_DIR.glob("split_selftest*"):
            f.unlink()

    print("\n================ SELF-TEST SUMMARY ================")
    for name, ok, sec in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({sec:.0f}s)")
    if all(ok for _, ok, _ in results):
        print("\nALL CHECKS PASSED - the project works on this computer.")
    else:
        print("\nSome checks failed - send a screenshot of the first FAILED step.")
        sys.exit(1)


if __name__ == "__main__":
    main()
