"""
Winter Road Surface Classifier — training script.
Team Surface 3, AI Implementation Technologies.

Classifies road-facing camera frames as: dry, wet, snow, ice.

Expected data layout — the session directory is REQUIRED, because the
train/val/test split is made at session level, not frame level:

    data/frames/
        20261015_routeA_morning/      <- one recording session
            dry/   frame_0001.jpg ...
            wet/   frame_0043.jpg ...
        20261016_routeB_evening/
            snow/  frame_0001.jpg ...
            ice/   frame_0120.jpg ...

Usage:
    python train.py --data_root data/frames --epochs 25
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["dry", "wet", "snow", "ice"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Tie-breaking noise in the split search. Large enough that a different seed
# gives a different split, small enough that a clearly better choice still
# wins. With few sessions, run several seeds and report mean and standard
# deviation rather than one number.
JITTER = 0.05


# --------------------------------------------------------------------------
# Data manifest and session-level splitting.
# Kept free of torch so it can be unit-tested on its own.
# --------------------------------------------------------------------------

def build_manifest(data_root):
    """Walk data_root/<session>/<class>/*.jpg -> [(path, label, session), ...]."""
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    manifest = []
    for session_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
            label = class_dir.name.lower()
            if label not in CLASSES:
                print(f"  ! skipping unknown class directory: {class_dir}")
                continue
            for img in sorted(class_dir.iterdir()):
                if img.suffix.lower() in IMG_EXTS:
                    manifest.append((str(img), label, session_dir.name))

    if not manifest:
        raise RuntimeError(f"No images found under {data_root}")
    return manifest


def split_by_session(manifest, ratios=(0.70, 0.15, 0.15), seed=42):
    """
    Partition whole sessions into train/val/test.

    Splitting by frame would scatter near-duplicate frames from the same
    drive across all three sets, so the test score would mostly measure
    memorisation of scenes the model had already seen. Every frame from a
    session therefore lands in exactly one split.

    Class coverage is reserved before frame counts are balanced. A drive is
    often all one condition, so filling the splits by size alone tends to
    leave val or test with no ice frames at all — which silently removes the
    class the project exists to detect from the evaluation.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"

    by_session = defaultdict(list)
    for row in manifest:
        by_session[row[2]].append(row)

    sessions = sorted(by_session)
    if len(sessions) < 3 * len(CLASSES):
        print(f"  ! only {len(sessions)} sessions for {len(CLASSES)} classes. "
              f"Aim for at least {3 * len(CLASSES)} separate drives so every "
              f"split can see every condition.")

    session_counts = {s: Counter(r[1] for r in rows)
                      for s, rows in by_session.items()}
    size = {s: len(rows) for s, rows in by_session.items()}
    class_total = Counter(r[1] for r in manifest)

    rng = random.Random(seed)
    tie = {s: rng.random() for s in sessions}

    total = len(manifest)
    targets = [r * total for r in ratios]
    buckets = [[], [], []]
    counts = [0, 0, 0]
    cls_counts = [Counter(), Counter(), Counter()]
    unassigned = set(sessions)

    def assign(session, i):
        buckets[i].append(session)
        counts[i] += size[session]
        cls_counts[i].update(session_counts[session])
        unassigned.discard(session)

    sessions_with = {c: sum(1 for s in sessions if session_counts[s][c])
                     for c in CLASSES}

    def overshoot(session, i):
        """How far adding this session pushes split i past its share of any
        class, summed and normalised. A session that satisfies the class we
        need while flooding the split with a different one is a bad pick."""
        cost = 0.0
        for c in CLASSES:
            n_c = class_total.get(c, 0)
            if not n_c:
                continue
            excess = cls_counts[i][c] + session_counts[session][c] - ratios[i] * n_c
            if excess > 0:
                cost += excess / n_c
        return cost

    # Pass 1 — give val and test one session of each class outright. Dealing
    # purely in proportion sends the earliest sessions to train, whose share
    # is largest, and a class with only a handful of sessions can run out
    # before test is reached. The smallest qualifying session is chosen so
    # that the bulk of a rare class still ends up in train. Classes are
    # handled rarest first, while there is still a choice of sessions.
    for cls in sorted(CLASSES, key=lambda c: (sessions_with[c], c)):
        if sessions_with[cls] < 3:
            if sessions_with[cls]:
                print(f"  ! only {sessions_with[cls]} session(s) contain "
                      f"'{cls}'. It cannot appear in all three splits.")
            continue
        for i in (1, 2):
            if cls_counts[i][cls]:
                continue
            candidates = [s for s in unassigned if session_counts[s][cls]]
            if len(candidates) <= 1:
                break  # leave the last one for train
            # Jitter lets near-equal candidates swap between seeds, so that
            # re-running with several seeds gives genuinely different splits.
            scored = [(overshoot(s, i) + rng.uniform(0, JITTER), tie[s], s)
                      for s in candidates]
            assign(min(scored)[2], i)

    # Pass 2 — deal what is left in proportion to the ratios, rarest class
    # first, because the rare class is the binding constraint: once its few
    # sessions are placed, the common classes still have plenty of sessions
    # left to balance with. Within a class the biggest contributors go first,
    # so the last few sessions make small corrections rather than large ones.
    for cls in sorted(CLASSES, key=lambda c: (class_total.get(c, 0), c)):
        n_cls = class_total.get(cls, 0)
        if not n_cls:
            continue
        candidates = sorted((s for s in unassigned if session_counts[s][cls]),
                            key=lambda s: (-session_counts[s][cls], tie[s]))
        for session in candidates:
            if session not in unassigned:
                continue

            def deficit(k):
                # Primary term: how far this split is below its share of the
                # class. Secondary term keeps overall split sizes sane.
                by_class = (ratios[k] * n_cls - cls_counts[k][cls]) / n_cls
                by_frames = (targets[k] - counts[k]) / max(total, 1)
                return by_class + 0.3 * by_frames

            jittered = [(deficit(k) + rng.uniform(0, JITTER), k)
                        for k in range(3)]
            assign(session, max(jittered)[1])

    # Any session whose classes were all exhausted above (defensive only).
    for session in sorted(unassigned, key=lambda s: (-size[s], tie[s])):
        assign(session, max(range(3), key=lambda k: targets[k] - counts[k]))

    # Pass 3 — with very few sessions the passes above can hand everything to
    # train, leaving nothing to validate or test on. Move one session back,
    # choosing the one whose departure does not remove a class from train
    # outright. This produces a poor split, not a good one; the warnings
    # above are the real answer, and the fix is more separate drives.
    for i in (1, 2):
        if buckets[i] or len(buckets[0]) < 3:
            continue
        def removal_cost(s):
            strands = sum(1 for c in CLASSES if session_counts[s][c]
                          and cls_counts[0][c] == session_counts[s][c])
            return (strands, size[s], tie[s])
        session = min(buckets[0], key=removal_cost)
        buckets[0].remove(session)
        counts[0] -= size[session]
        cls_counts[0].subtract(session_counts[session])
        assign(session, i)

    splits = {}
    for name, bucket in zip(("train", "val", "test"), buckets):
        rows = [r for s in bucket for r in by_session[s]]
        rng.shuffle(rows)
        splits[name] = rows

        missing = set(CLASSES) - {r[1] for r in rows}
        if missing:
            print(f"  ! {name} split has no {sorted(missing)} frames — no "
                  f"session containing them was left to allocate. Collect "
                  f"more separate sessions for those conditions.")

    return splits


def describe(splits):
    """Print frames, share of total, sessions, and the class breakdown."""
    total = sum(len(r) for r in splits.values())
    print(f"\n{'split':<8}{'frames':>8}{'share':>8}{'sessions':>10}"
          f"   class distribution")
    print("-" * 74)
    for name, rows in splits.items():
        counts = Counter(r[1] for r in rows)
        dist = "  ".join(f"{c}:{counts.get(c, 0)}" for c in CLASSES)
        share = f"{100 * len(rows) / total:.0f}%" if total else "-"
        n_sessions = len({r[2] for r in rows})
        print(f"{name:<8}{len(rows):>8}{share:>8}{n_sessions:>10}   {dist}")
    print("\nShares drift from 70/15/15 because whole sessions are "
          "indivisible. Report the actual split in your write-up.\n")


# --------------------------------------------------------------------------
# Everything below needs torch. The import is guarded so that build_manifest
# and split_by_session above stay importable and testable without it.
# --------------------------------------------------------------------------

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:  # --dry_run still works
    Dataset = object
    TORCH_AVAILABLE = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RoadCrop:
    """Discard sky, bonnet and dashboard before resizing, so the 224x224 the
    network sees is mostly road surface. Tune the fractions to your mount
    position by eye on a handful of frames, then keep them fixed for the
    whole project — changing them mid-way invalidates earlier results."""

    def __init__(self, top=0.45, side=0.15):
        self.top, self.side = top, side

    def __call__(self, img):
        w, h = img.size
        return img.crop((int(w * self.side), int(h * self.top),
                         int(w * (1 - self.side)), h))


class FrameDataset(Dataset):
    """Defined at module level so DataLoader workers can pickle it."""

    def __init__(self, rows, tf):
        self.rows, self.tf = rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image
        path, label, _ = self.rows[i]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.tf(img), CLASSES.index(label)


def run_training(args):
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Install it with:\n"
            "    pip install torch torchvision scikit-learn\n"
            "or run on Google Colab, where it is preinstalled. "
            "Use --dry_run to inspect the split without training.")

    import numpy as np
    import torch.nn as nn
    from sklearn.metrics import classification_report, confusion_matrix
    from torch.utils.data import DataLoader
    from torchvision import models, transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -- Transforms -------------------------------------------------------
    # Colour jitter is deliberately mild: sheen, specular highlights and
    # colour temperature are much of what separates wet from ice, and strong
    # jitter destroys exactly that signal.
    train_tf = transforms.Compose([
        RoadCrop(args.crop_top, args.crop_side),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        RoadCrop(args.crop_top, args.crop_side),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # -- Data -------------------------------------------------------------
    manifest = build_manifest(args.data_root)
    splits = split_by_session(manifest, seed=args.seed)
    describe(splits)

    for name in ("train", "val", "test"):
        if not splits[name]:
            raise RuntimeError(
                f"The {name} split is empty. There are too few recording "
                f"sessions to partition. Collect more separate drives, then "
                f"re-run with --dry_run to check the split before training.")

    loaders = {
        name: DataLoader(
            FrameDataset(rows, train_tf if name == "train" else eval_tf),
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
        )
        for name, rows in splits.items()
    }

    # -- Model ------------------------------------------------------------
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(device)

    # Ice is the rarest class and the most costly to miss, so weight the loss
    # by inverse class frequency rather than training on raw counts.
    train_counts = Counter(r[1] for r in splits["train"])
    weights = torch.tensor(
        [len(splits["train"]) / (len(CLASSES) * max(train_counts.get(c, 0), 1))
         for c in CLASSES],
        dtype=torch.float, device=device,
    )
    print("class weights: " +
          ", ".join(f"{c}={w:.2f}" for c, w in zip(CLASSES, weights.tolist())))
    criterion = nn.CrossEntropyLoss(weight=weights)

    def evaluate(loader):
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in loader:
                out = model(x.to(device, non_blocking=True))
                y_pred += out.argmax(1).cpu().tolist()
                y_true += y.tolist()
        return y_true, y_pred

    def macro_f1(y_true, y_pred):
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average="macro", zero_division=0)

    def train_phase(phase, n_epochs, lr, best_f1, frozen_backbone=False):
        if n_epochs <= 0:
            return best_f1
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

        for epoch in range(1, n_epochs + 1):
            model.train()
            if frozen_backbone:
                # requires_grad=False stops the weights moving but NOT the
                # BatchNorm running statistics, which model.train() keeps
                # updating from our data. Left alone, the pretrained features
                # drift during a phase that is supposed to freeze them.
                for m in model.modules():
                    if isinstance(m, nn.modules.batchnorm._BatchNorm):
                        m.eval()
            running = 0.0
            for x, y in loaders["train"]:
                x, y = x.to(device, non_blocking=True), y.to(device)
                opt.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                opt.step()
                running += loss.item() * x.size(0)
            sched.step()

            f1 = macro_f1(*evaluate(loaders["val"]))
            flag = ""
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), args.out)
                flag = "  <- saved"
            print(f"[{phase}] epoch {epoch:>3}/{n_epochs}  "
                  f"train_loss {running / len(splits['train']):.4f}  "
                  f"val_macroF1 {f1:.4f}{flag}")
        return best_f1

    # Phase 1 trains only the new head, so the randomly initialised layer does
    # not push large gradients back through the pretrained features.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    best = train_phase("head", args.warmup_epochs, args.lr_head, 0.0,
                       frozen_backbone=True)

    # Phase 2 fine-tunes the whole network at a much lower rate.
    for p in model.parameters():
        p.requires_grad = True
    best = train_phase("full", args.epochs, args.lr_full, best)

    # -- Test -------------------------------------------------------------
    model.load_state_dict(
        torch.load(args.out, map_location=device, weights_only=True))
    y_true, y_pred = evaluate(loaders["test"])

    print(f"\nbest val macro-F1: {best:.4f}")
    print("\nTest set — held-out sessions\n")
    print(classification_report(y_true, y_pred, labels=range(len(CLASSES)),
                                target_names=CLASSES, digits=3,
                                zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    print("confusion matrix (rows = true, cols = predicted)")
    print(f"{'':>8}" + "".join(f"{c:>8}" for c in CLASSES))
    for name, row in zip(CLASSES, cm):
        print(f"{name:>8}" + "".join(f"{v:>8}" for v in row))

    # The wet/ice confusion is the one that matters operationally: ice read as
    # wet means a hazardous road is never treated.
    i_ice, i_wet = CLASSES.index("ice"), CLASSES.index("wet")
    n_ice = int(cm[i_ice].sum())
    if n_ice:
        print(f"\nice misread as wet: {cm[i_ice][i_wet]}/{n_ice} "
              f"({100 * cm[i_ice][i_wet] / n_ice:.1f}% of ice frames)")

    Path("results.json").write_text(json.dumps({
        "best_val_macro_f1": best,
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
        "split_sessions": {k: sorted({r[2] for r in v})
                           for k, v in splits.items()},
    }, indent=2))
    print("\nwrote results.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/frames")
    p.add_argument("--out", default="best_model.pt")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_full", type=float, default=1e-4)
    p.add_argument("--crop_top", type=float, default=0.45,
                   help="fraction of frame height removed from the top")
    p.add_argument("--crop_side", type=float, default=0.15,
                   help="fraction of frame width removed from each side")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true",
                   help="build the manifest and split only, no training")
    args = p.parse_args()

    if args.dry_run:
        describe(split_by_session(build_manifest(args.data_root),
                                  seed=args.seed))
        return
    run_training(args)


if __name__ == "__main__":
    main()
