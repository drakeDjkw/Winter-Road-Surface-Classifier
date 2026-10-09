# Winter-Road-Surface-Classifier
Team Surface 3 — Yikwa Sandrakh, Wenda Waipo, Agata Melani Artificial Intelligence Implementation Technologies, Baikal School of BRICS, INRTU

## Project direction

For road maintenance dispatchers, we want to change the lack of real-time road-surface visibility so that icy and snow-covered sections are identified and prioritised for treatment faster.

## What the system does

Classifies road-facing camera frames into one of four surface states — **dry, wet, snow-covered, icy** — so that dispatchers can see which stretches need urgent treatment rather than relying on manual inspection or driver reports.

Single-label classification, one verdict per frame. Not object detection: surface condition covers the whole visible road and there is no object to localise. Section "Decisions" explains how that was settled.

## Status

| | |
|---|---|
| Footage recorded |
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
3. The imagery was stock dashcam footage — Montevideo street scenes, a palm-lined road at sunset, a US interstate — not Irkutsk.
4. Annotation boxes spanned sky, trees and oncoming traffic, with widths clustered at 1.0.

Cause 4 is what settled the task framing: annotators were drawing whole-frame verdicts, which is a classification label expressed as a box.

## Pipeline

| Stage | Tool | Output |
|---|---|---|
| Frame extraction from video | `extract_frames.py` | `<session>/<class>/` frames |
| Annotation (optional) | Roboflow | labelled export |
| Regroup an export by session | `roboflow_to_sessions.py` | `<session>/<class>/` frames |
| Split, train, evaluate | `winter_road_surface.ipynb` | weights, metrics, confusion matrix |

```bash
# 1. video -> labelled frames


# 2. or, if the frames were labelled in Roboflow


# 3. check the split before training anything


# 4. zip and train 

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

- **Source:** road-facing footage, phone-mounted, recorded while driving in Irkutsk
- **Classes:** dry, wet, snow-covered, icy
- **Target:** ~1,000 frames per class (600 minimum), sampled at one frame per three seconds
- **Sessions:** 8–10 distinct drives per class, varied by date, route, time of day and lighting
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

## Repository


## Open questions
