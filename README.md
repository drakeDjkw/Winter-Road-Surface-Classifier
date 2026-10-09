# Winter Road Surface Classifier

**Team Surface 3** — Yikwa Sandrakh, Wenda Waipo, Agata Melani
Artificial Intelligence Implementation Technologies · Baikal School of BRICS, INRTU

## Project direction

For road maintenance dispatchers, we want to change the lack of real-time road-surface visibility so that icy and snow-covered sections are identified and prioritised for treatment faster.

## What the system does

Classifies road-facing camera frames into one of four surface states — **dry, wet, snow-covered, icy** — so that dispatchers can see which stretches need urgent treatment rather than relying on manual inspection or driver reports.

Single-label classification, one verdict per frame. Not object detection: surface condition covers the whole visible road and there is no object to localise. Section "Decisions" explains how that was settled.

## Status

| | |
|---|---|
| Footage recorded | 20 drives (independence not yet verified) |
| Frames labelled | 256 |
| Training runs | 2, both discarded — see below |
| Usable result | none yet |

The pipeline runs end to end. No trustworthy performance figure exists yet, and none should be quoted.

### Discarded runs

Two models were trained during exploration. Neither is used, and neither one's metrics belong in a report.

**`surface-fn4sy-1-yolo26n-t1`** (Roboflow, YOLO26n). Object detection, trained on Roboflow's random frame-level split. Wrong task, and the split leaked.

**`runs/detect/train`** (local, YOLO11n). Object detection, 3 epochs. Reported mAP@50 of 0.371 alongside precision of 0.00485 — the model emitted roughly 7,600 false-positive boxes against 45 true objects in validation. Four causes, all diagnosed:

1. `epochs: 3` with `warmup_epochs: 3.0` — the learning rate never left warmup, peaking at 0.00081 against a target of 0.01. Total training time was 43 seconds.
2. 221 annotated instances across four classes, of which **wet had 8**. Validation contained one wet object.
3. Annotation boxes spanned sky, trees and oncoming traffic, with widths clustered at 1.0.
4. The run therefore says nothing about Irkutsk conditions specifically — see **Data** for the dataset scope and its limitation.

Cause 3 is what settled the task framing: annotators were drawing whole-frame verdicts, which is a classification label expressed as a box.

## Pipeline

| Stage | Tool | Output |
|---|---|---|
| Frame extraction from video | `extract_frames.py` | `<session>/<class>/` frames |
| Annotation (optional) | Roboflow | labelled export |
| Regroup an export by session | `roboflow_to_sessions.py` | `<session>/<class>/` frames |
| Split, train, evaluate | `winter_road_surface_colab.ipynb` | weights, metrics, confusion matrix |

```bash
# 1. video -> labelled frames
python extract_frames.py scan    --videos ./footage     # writes labels.csv
python extract_frames.py report  --videos ./footage     # check thresholds
python extract_frames.py extract --videos ./footage --out ./frames

# 2. or, if the frames were labelled in Roboflow
python roboflow_to_sessions.py --export ./roboflow-export --out ./frames --write

# 3. check the split before training anything
python train.py --dry_run --data_root ./frames

# 4. zip and train in Colab
zip -r frames.zip frames
```

### Session-level splitting

**The split is by recording session, never by frame.** Frames sampled from one drive are near-duplicates. A random frame-level split puts near-identical frames in both training and test, so the test score measures memorisation rather than generalisation.

Measured on a 16-session dataset of this shape:

| Split method | Test frames from a session the model trained on |
|---|---|
| Frame-level (random) | 39 / 39 — **100%** |
| Session-level (ours) | 0 / 58 — **0%** |

Every test frame in the naive split comes from a drive the model already saw. Section 10 of the notebook trains the same model under both splits so the cost can be reported as a number.

The splitter reserves one session of each class for validation and for test before balancing sizes — otherwise a class appearing in only a few drives (ice, always) lands entirely in training and vanishes from the evaluation.

## Data

**Source: public dashcam footage**, spanning several countries, climates and camera setups. This is a deliberate scope choice. A full range of winter conditions — icy roads above all — cannot be recorded to a semester timetable, and public video already covers variety we could not reach by driving.

**The limitation, stated rather than hidden:** domain shift. Camera, mount height, climate, road markings and traffic conventions all differ from Irkutsk, so accuracy measured on this dataset does not transfer to local roads unmeasured. Locally recorded footage, once collected, becomes a fine-tuning set on top of this one rather than a replacement for it.

- **Classes:** dry, wet, snow-covered, icy
- **Target:** ~1,000 frames per class (600 minimum), sampled at one frame per three seconds
- **Sessions:** 8–10 distinct source videos per class, varied by location, time of day and lighting. One source video is one session.
- **Split:** 70/15/15 by session; realised proportions drift because sessions are indivisible, so the actual figures are reported
- **Quality filtering:** blurred frames and near-duplicates from stationary stretches are dropped, with thresholds derived from the footage rather than fixed in advance
- **Class balance:** ice will be under-represented; class weighting is applied and per-class metrics reported instead of accuracy

## Model

- **Architecture:** ResNet-18, pretrained on ImageNet, fine-tuned. A small CNN trained from scratch serves as a baseline to quantify what the pretrained features contribute.
- **Framework:** PyTorch with torchvision
- **Input:** cropped to the road region, resized to 224 × 224, ImageNet normalisation. Augmentation limited to horizontal flip and mild colour jitter — sheen and colour temperature carry much of the wet/icy signal, and aggressive jitter destroys it.
- **Training:** 3 epochs head-only with the backbone frozen, then ~20 fine-tuning the whole network. AdamW at 1e-3 then 1e-4, batch 32, cosine annealing, class-weighted cross-entropy. BatchNorm layers are held in eval mode during the frozen phase, since disabling gradients alone does not stop their running statistics drifting.
- **Hardware:** Google Colab, NVIDIA T4

## Evaluation

Macro-averaged F1 is the headline metric, with per-class precision and recall and a 4×4 confusion matrix. Accuracy is **not** the primary metric — it is dominated by the majority classes and hides failure on ice.

The ice→wet error rate is reported separately: an icy road read as merely wet never gets treated, which is the error with a safety cost attached.

Because the number of sessions is small, results are reported as a mean and standard deviation across several session splits, not as a single figure.

## Decisions

**Classification, not detection.** Surface condition is a property of the whole visible road, so there is nothing for a box to enclose. Detection would cost box annotation on every frame and would not improve what a dispatcher can act on — they need a condition per road section, not image coordinates. The annotation statistics from the discarded YOLO run confirmed this: box widths clustered at 1.0, meaning whole-frame labels drawn as boxes.

**Roboflow for annotation only.** Its training facility and its automatic train/validation/test division are not used — that division operates at frame level and reintroduces the leakage above. Training happens in this repository's notebook, where the partitioning is explicit and reproducible.

## Delivery

How a model like this reaches the dispatcher it is for:

```
camera feed  →  inference service  →  label + confidence  →  dispatcher view
(sampled every     (ResNet-18 as ONNX,    {camera_id, timestamp,    (sections ranked
 few minutes)       one CPU box)           class, probability}       by hazard)
```

**The latency budget is minutes, not milliseconds.** Treatment rounds are planned on a timescale of tens of minutes, so batched inference over the camera feeds is sufficient. No real-time engineering, and no GPU at inference for ResNet-18 at 224 × 224.

**Abstain rather than guess.** Below a confidence threshold the service returns `uncertain` instead of a class. A confident wrong answer about ice is worse than no answer, which is the same reasoning behind reporting the ice→wet error rate separately.

**Corrections feed back.** Every prediction is logged. What a dispatcher overrides becomes labelled training data, which is the cheapest source of new labels there is.

## Where the accuracy comes from

Ordered by return per unit of effort, not by sophistication.

| # | Lever | Why it pays | Effort |
|---|---|---|---|
| 1 | More wet and icy examples | 8 wet instances cannot be learned; this is the binding constraint | Weeks of collection |
| 2 | A written labelling guide | No model exceeds the consistency of its labels | One team session |
| 3 | Train past warmup | Nothing converged in 43 seconds | Free |
| 4 | Temporal smoothing | Consecutive frames of one drive should agree; a majority vote over a short window removes flicker | Free — the data is video |
| 5 | Fine-tune on local footage | Closes the domain gap between public video and Irkutsk | Once local data exists |
| 6 | Larger backbone | ResNet-50 helps only once the dataset can support it | Minutes, but premature now |
| 7 | Calibration and abstention | Turns an uncertain prediction into a useful "unknown" | Low |

Rows 2, 3 and 4 cost nothing and come before any change of architecture. Row 4 is the one most easily missed: because the frames come from continuous video, neighbouring frames are nearly the same scene and should receive the same label, so averaging predictions over a short window buys accuracy with no extra data and no extra training.

## Repository

```
README.md
docs/project_brief.md
extract_frames.py              # video -> labelled frames, with quality filtering
roboflow_to_sessions.py        # Roboflow export -> session-grouped frames
train.py                       # session-level split + training (CLI)
winter_road_surface_colab.ipynb
requirements.txt
```

## Open questions

- Final choice of user: dispatchers (prioritising treatment) or drivers (real-time warnings). This repository proceeds on the dispatcher assumption, not yet confirmed by the team.
- How many source videos contain genuine icy conditions, which is the binding constraint on the whole project.
- The gap between 221 annotated instances and the ~4,000-frame target, and the 8 wet examples in particular.
- Crop fractions for the road region, which vary by source video rather than being fixed by one mount.
- Whether any locally recorded footage will be added as a fine-tuning set this winter.
