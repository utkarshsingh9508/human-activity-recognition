# Real-Time Skeleton-Based Human Activity Recognition (ST-GCN) — Final

Vision-based HAR: camera → **MediaPipe Pose** (33 joints, x/y/z) → **ST-GCN** graph network →
activity, in real time on a CPU, plus an AI fitness coach. Highlights:

| Feature | Why it matters |
|---|---|
| **Cross-subject testing** (`--split xsub`) | test on a person the model has never seen, as in the literature |
| **Ablation study** (`ablation.py`) | one table showing what each component contributes |
| **NTU RGB+D 60 benchmark** (`ntu_convert.py`, `--profile ntu`) | compare against the published ST-GCN numbers |
| **Two-stream (joint + bone)** (`--stream bone`, `--fuse`) | the 2s-AGCN idea, usually a clear accuracy gain |
| **Adaptive graph** (`--model agcn`) | learns links that are not bones (e.g. hand–hand) |
| **Speed augmentation** | robust to people doing actions faster or slower |
| **Apple GPU (MPS)** | automatic on M-series Macs, several times faster training |
| **Clip accuracy** | window scores averaged per recording, the number papers report |

## Quick start on macOS (final version)

```bash
cd ~/Downloads/skeleton_har_final
bash setup_mac.sh
source venv/bin/activate
python selftest.py
```
`setup_mac.sh` creates the environment and installs everything (once).
`selftest.py` runs the whole pipeline on a small synthetic dataset in ~2–5 minutes and must
end with **ALL CHECKS PASSED**. Then record real data (section 1) and continue.

Every new Terminal: `cd ~/Downloads/skeleton_har_final` then `source venv/bin/activate`.

If you already recorded clips in another copy of the project, copy them over:
`cp -R <old folder>/data/keypoints ~/Downloads/skeleton_har_final/data/`

## 1. Record data – now with the person's name

```bash
python collect_data.py --subject utkarsh --action waving --clips 40
```
All actions for one person:
```bash
for a in standing walking sitting waving clapping squatting jumping_jacks idle; do python collect_data.py --subject utkarsh --action $a --clips 40; done
```
Then the second person does the same with `--subject <their name>`. Check:
```bash
python set_subject.py --list
```

## 2. Train

```bash
python train.py                                   # ST-GCN, random split  -> checkpoints/stgcn.pt
python train.py --split xsub --test-subjects priya   # cross-subject       -> stgcn_xsub.pt
python train.py --stream bone                     # bone stream           -> stgcn_bone.pt
python train.py --model agcn                      # adaptive graph        -> agcn.pt
python train.py --model cnn_lstm                  # baseline              -> cnn_lstm.pt
```
Device is picked automatically (Apple GPU `mps` on your Mac). Add `--split xsub
--test-subjects <name>` to any command for the cross-subject version.

## 3. Evaluate

```bash
python evaluate.py --model checkpoints/stgcn.pt checkpoints/agcn.pt checkpoints/cnn_lstm.pt
python evaluate.py --model checkpoints/stgcn.pt checkpoints/stgcn_bone.pt --fuse   # two-stream
```
Reports window accuracy, clip accuracy, macro-F1, CPU latency and size; saves reports and
confusion matrices in `results/`.

## 4. Ablation study (one command)

```bash
python ablation.py --split xsub --test-subjects priya --epochs 40
```
Trains every variant and writes `results/ablation_webcam_xsub_priya.md`:

| Variant | what is changed |
|---|---|
| ST-GCN (full) | reference |
| − z coordinate | 2D only |
| − velocity | positions only, no motion channels |
| − virtual joints | no mid-hip/neck, nose becomes the graph centre |
| distance / uniform partitioning | K=2 / K=1 instead of the spatial K=3 |
| − edge importance | fixed skeleton weights |
| − speed augmentation | no temporal resampling |
| bone stream, two-stream fusion | 2s-AGCN idea |
| AGCN | adaptive graph |
| CNN-LSTM | no graph at all |

`--seeds 3` repeats each run with 3 seeds and reports mean ± std (more reliable, 3× the
time). Finished runs are reused, so you can stop and resume. `--only full no_z bone two_stream`
runs a subset.

**How to present it:** differences smaller than the ± std are not meaningful. With a small
self-recorded dataset some rows may be within noise, which is a valid finding to report.

## 5. NTU RGB+D 60 benchmark (optional, big)

1. Request access: https://rose1.ntu.edu.sg/dataset/actionRecognition/ and download
   `nturgbd_skeletons_s001_to_s017.zip` (skeleton files only).
2. Convert (single-person skeletons, 25 Kinect joints):
   ```bash
   python ntu_convert.py --src path/to/nturgb+d_skeletons --workers 8
   ```
3. Train with the official protocols:
   ```bash
   python train.py --profile ntu --split xsub --workers 4
   python train.py --profile ntu --split xview --workers 4
   ```
Reference: ST-GCN reports 81.5 % (X-Sub) and 88.3 % (X-View). This implementation keeps
one person per clip (the paper models two), uses 64 resampled frames, and fewer epochs by
default, so expect a somewhat lower number; state these differences in your report.
Training takes hours even on the M3 GPU - start it overnight.

## 6. Edge export, explainability, live demo

```bash
python export_edge.py --ckpt checkpoints/stgcn.pt            # ONNX + INT8 + benchmark
python visualize_graph.py --ckpt checkpoints/agcn.pt         # learned links, incl. non-bones
python realtime.py --model checkpoints/stgcn.pt
python realtime.py --model checkpoints/stgcn.pt checkpoints/stgcn_bone.pt   # two-stream live
python realtime.py --model exports/stgcn_int8.onnx --complexity 0          # edge mode
```

## 7. AI fitness coach (sports use case)

Counts reps and gives form feedback in real time for **squats** and **jumping jacks**.

```bash
python sports_coach.py --exercise squat
python sports_coach.py --exercise jumping_jack --voice        # spoken feedback on macOS
python sports_coach.py --exercise auto --model checkpoints/stgcn.pt
```
Keys: `q` quit · `r` reset · `1` squat · `2` jumping jack · `a` auto-detect.
When you quit, a report is saved in `results/sessions/`: a JSON file with every rep and a
chart of the knee/arm angle over time (green = clean rep, orange = form issue, red = not counted).

**How it works – a hybrid of learning and geometry**
- **ST-GCN** (auto mode) recognises *which* exercise is being done. For this, record
  `squatting` and `jumping_jacks` clips with `collect_data.py` and retrain.
- **Joint-angle geometry** on MediaPipe *world* landmarks (3D, metres) counts reps with a
  hysteresis state machine and judges each rep:

| Exercise | Rep counted when | Form checks |
|---|---|---|
| Squat | knee angle goes below 140°, then back above 160°; **valid** only if the deepest knee angle ≤ 100° | depth ("Go lower"), torso lean > 50° ("Keep your chest up"), knees caving in, front view only ("Push your knees out"), rep < 1 s ("Slow down") |
| Jumping jack | arms/legs open, then arms down and feet together; **valid** only if arms reach ≥ 145° and feet ≥ 2× hip width | arms not overhead, feet not wide enough, feet not brought back together, arms and legs out of sync |

All thresholds are at the top of `exercises.py` (`SQUAT`, `JACK`) and can be tuned.
Tips: side-on to the camera is best for squat depth and lean; facing the camera is needed for
the knee-cave check and for jumping jacks. Keep the whole body in view.

## Files

| File | Role |
|---|---|
| `config.py` | paths, dataset profiles (webcam / ntu), training defaults |
| `layouts.py` | joint topologies: MediaPipe (±virtual joints), NTU Kinect |
| `features.py` | normalisation, 3D augmentation, bone vectors, feature switches |
| `dataset.py` | random / cross-subject / cross-view splits, windowing, speed augmentation |
| `models/graph.py` | uniform, distance and spatial partitioning |
| `models/stgcn.py` | ST-GCN + adaptive graph (AGCN) |
| `models/cnn_lstm.py` | baseline |
| `train.py`, `evaluate.py`, `ablation.py` | experiments |
| `export_edge.py`, `inference.py`, `realtime.py` | deployment |
| `collect_data.py`, `extract_from_videos.py`, `set_subject.py`, `ntu_convert.py` | data |
| `exercises.py`, `sports_coach.py` | rep counting, form feedback, fitness-coach app |
| `setup_mac.sh`, `selftest.py`, `synth.py` | one-step setup and full self-test on synthetic data |

References: Yan et al., ST-GCN, AAAI 2018 · Shi et al., 2s-AGCN, CVPR 2019 ·
Shahroudy et al., NTU RGB+D, CVPR 2016.
