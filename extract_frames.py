"""
Video -> labelled frames, for the winter road surface classifier.
Team Surface 3.

Turns recorded drives into the directory layout the training pipeline expects:

    frames/<session>/<class>/<session>_000123.jpg

Three steps:

  1. Build a labelling sheet listing every video found:
         python extract_frames.py scan --videos ./footage

  2. Open labels.csv and fill in the 'label' column. One row per video is
     enough when a drive is all one condition; add extra rows with start/end
     times where the condition changes part-way.

  3. Check the quality thresholds against your own footage, then extract:
         python extract_frames.py report  --videos ./footage
         python extract_frames.py extract --videos ./footage --out ./frames

Frames are saved uncropped. Cropping happens at training time, so the crop can
be re-tuned later without re-extracting everything.
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["dry", "wet", "snow", "ice"]
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts", ".3gp"}
CSV_FIELDS = ["video", "session", "label", "start", "end", "notes"]


# --------------------------------------------------------------------------
# Quality scoring
# --------------------------------------------------------------------------

def road_region(frame, crop_top, crop_side):
    """The part of the frame the model will actually see."""
    h, w = frame.shape[:2]
    return frame[int(h * crop_top):, int(w * crop_side):int(w * (1 - crop_side))]


def sharpness(gray):
    """Variance of the Laplacian. Low means blurred — motion blur, a dirty
    windscreen, or a wiper mid-sweep."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def otsu_split(values, bins=96):
    """Otsu's method: the threshold maximising between-class variance.

    Blurred and stationary stretches form a low cluster separate from normal
    driving, so the job is to find the valley between two groups — which is
    what Otsu does. A fraction of the median cannot: here the static cluster
    sat at 1.6 against a median of 3.6, and any fixed fraction either lands
    inside the driving cluster or below the static one.
    """
    values = np.asarray(values, dtype=float)
    hist, edges = np.histogram(values, bins=bins)
    total = hist.sum()
    if total == 0:
        return None

    p = hist / total
    centres = (edges[:-1] + edges[1:]) / 2
    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    sigma_b[~np.isfinite(sigma_b)] = 0.0

    k = int(np.argmax(sigma_b))
    threshold = float(edges[k + 1])

    low, high = values[values < threshold], values[values >= threshold]
    if low.size == 0 or high.size == 0:
        return None
    return {
        "threshold": threshold,
        "drop_rate": 100.0 * low.size / values.size,
        "low_mean": float(low.mean()),
        "high_mean": float(high.mean()),
    }


def suggest_threshold(values, max_drop=40.0, max_ratio=0.6):
    """Otsu's split, accepted only when it reflects a real gap.

    Otsu always returns something: on uniformly good footage it would bisect
    a single cluster and throw away half the frames. It is therefore accepted
    only when the two groups are genuinely far apart and the discard stays
    modest. Otherwise no filtering is suggested, which is the right answer for
    footage with no blurred or stationary stretches.
    """
    split = otsu_split(values)
    if split is None:
        return 0.0, None, "not enough data"

    ratio = split["low_mean"] / max(split["high_mean"], 1e-9)
    if ratio > max_ratio:
        return 0.0, split, ("no distinct low cluster — the footage looks "
                            "uniform, so no filtering is needed")
    if split["drop_rate"] > max_drop:
        return 0.0, split, (f"would discard {split['drop_rate']:.0f}% of "
                            f"frames, too much to apply automatically")
    return split["threshold"], split, None


def difference(gray_small, previous):
    """Mean absolute difference from the last kept frame, on a thumbnail.
    Near zero means the vehicle was stationary and the frame adds nothing."""
    if previous is None:
        return 255.0
    return float(np.abs(gray_small.astype(np.int16)
                        - previous.astype(np.int16)).mean())


def iter_samples(path, interval, start=None, end=None):
    """Yield (timestamp, frame) every `interval` seconds.

    Frames are grabbed without decoding and only the wanted ones are decoded,
    which is far faster than decoding everything, and more reliable than
    seeking by timestamp on variable-frame-rate phone video.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:      # 0, or NaN
        fps = 30.0
        print(f"  ! {path.name}: no frame rate reported, assuming {fps:g} fps")

    step = max(1, int(round(fps * interval)))
    idx = 0
    try:
        while True:
            if not cap.grab():
                break
            t = idx / fps
            if idx % step == 0 and (start is None or t >= start) \
                    and (end is None or t <= end):
                ok, frame = cap.retrieve()
                if ok:
                    yield t, frame
            idx += 1
    finally:
        cap.release()


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def find_videos(video_dir):
    video_dir = Path(video_dir)
    if not video_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {video_dir}")
    return sorted(p for p in video_dir.rglob("*")
                  if p.suffix.lower() in VIDEO_EXTS)


def cmd_scan(args):
    """Write a labelling sheet listing every video found."""
    videos = find_videos(args.videos)
    if not videos:
        print(f"No video files under {args.videos}")
        return

    out = Path(args.labels)
    if out.exists() and not args.force:
        print(f"{out} already exists — not overwriting. Use --force to replace.")
        return

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for v in videos:
            cap = cv2.VideoCapture(str(v))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            mins = (n / fps / 60) if fps else 0
            w.writerow({"video": v.name, "session": v.stem, "label": "",
                        "start": "", "end": "",
                        "notes": f"{mins:.1f} min"})

    print(f"wrote {out} listing {len(videos)} videos\n")
    print("Next: open it and fill in the 'label' column with one of")
    print(f"  {', '.join(CLASSES)}\n")
    print("Leave a row's label blank to skip that video. If a drive changes")
    print("condition part-way, duplicate its row and set start/end in seconds:")
    print("  drive_03.mp4,drive_03,wet,0,240,")
    print("  drive_03.mp4,drive_03,snow,260,900,\n")
    print("IMPORTANT — the 'session' column is what the train/test split is")
    print("built on. Two clips from the same outing must share one session")
    print("name, or near-identical frames will end up on both sides of the")
    print("split and the test score will be meaningless.")


def read_labels(path):
    rows = []
    with Path(path).open(newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            label = (row.get("label") or "").strip().lower()
            if not label:
                continue
            if label not in CLASSES:
                print(f"  ! line {i}: unknown label {label!r}, skipping")
                continue
            def num(key):
                v = (row.get(key) or "").strip()
                return float(v) if v else None
            rows.append({
                "video": (row["video"] or "").strip(),
                "session": (row.get("session") or "").strip()
                           or Path(row["video"]).stem,
                "label": label,
                "start": num("start"),
                "end": num("end"),
            })
    return rows


def cmd_report(args):
    """Sample a few videos and print the distribution of quality scores, so the
    thresholds are chosen against this footage rather than guessed."""
    videos = find_videos(args.videos)[:args.sample_videos]
    if not videos:
        print(f"No video files under {args.videos}")
        return

    sharp, diffs = [], []
    for v in videos:
        prev = None
        n = 0
        for _, frame in iter_samples(v, args.interval):
            region = road_region(frame, args.crop_top, args.crop_side)
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (64, 64))
            sharp.append(sharpness(gray))
            if prev is not None:       # skip the first-frame sentinel
                diffs.append(difference(small, prev))
            prev = small
            n += 1
            if n >= args.sample_frames:
                break
        print(f"  sampled {n:>4} frames from {v.name}")

    def summary(name, values, unit=""):
        a = np.array(values)
        qs = np.percentile(a, [5, 25, 50, 75, 95])
        print(f"\n{name}")
        print(f"  5th {qs[0]:8.1f}{unit}   25th {qs[1]:8.1f}{unit}   "
              f"median {qs[2]:8.1f}{unit}")
        print(f"  75th {qs[3]:8.1f}{unit}   95th {qs[4]:8.1f}{unit}   "
              f"min {a.min():.1f}  max {a.max():.1f}")
        return qs

    print(f"\n{'=' * 64}\nQuality scores across {len(videos)} video(s)\n{'=' * 64}")
    summary("Sharpness (variance of Laplacian) — higher is sharper", sharp)
    summary("Difference from previous kept frame — low means stationary",
            diffs[1:] or diffs)

    sharp_a = np.array(sharp)
    diff_a = np.array(diffs[1:] or diffs)

    def histogram(values, width=44, bins=14):
        counts, edges = np.histogram(values, bins=bins)
        peak = counts.max() or 1
        print()
        for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
            bar = "#" * int(round(width * c / peak))
            print(f"  {lo:9.1f} – {hi:<9.1f} {bar} {c}")

    print(f"\n{'=' * 64}\nSuggested thresholds for this footage\n{'=' * 64}")
    for name, values, flag in (
            ("Sharpness", sharp_a, "--min_sharpness"),
            ("Frame-to-frame difference", diff_a, "--min_difference")):
        thr, split, reason = suggest_threshold(values)
        print(f"\n{name}")
        histogram(values)
        if reason:
            print(f"\n  {flag}: no threshold suggested — {reason}.")
        else:
            print(f"\n  {flag} {thr:.1f}"
                  f"   discards {split['drop_rate']:.1f}% of sampled frames")
            print(f"     low cluster mean {split['low_mean']:.1f}  vs  "
                  f"normal {split['high_mean']:.1f}")

    print(f"\n{'=' * 64}")
    print("Two clusters in a histogram means blurred or stationary stretches\n"
          "are present and worth removing. One cluster means the footage is\n"
          "uniform and no filtering is needed. `extract` computes these same\n"
          "thresholds automatically unless you pass your own.")


def sample_scores(videos, interval, crop_top, crop_side,
                  max_videos=4, max_frames=150):
    """Collect quality scores from a sample of the footage."""
    sharp, diffs = [], []
    for v in videos[:max_videos]:
        prev, n = None, 0
        for _, frame in iter_samples(v, interval):
            gray = cv2.cvtColor(road_region(frame, crop_top, crop_side),
                                cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (64, 64))
            sharp.append(sharpness(gray))
            # The first frame of each video has nothing to compare against and
            # scores a sentinel 255. Left in, those few values sit so far from
            # the rest that they become their own cluster and the real split
            # is missed entirely.
            if prev is not None:
                diffs.append(difference(small, prev))
            prev = small
            n += 1
            if n >= max_frames:
                break
    return np.array(sharp), np.array(diffs)


def resolve_thresholds(args, videos):
    """Turn 'auto' thresholds into numbers derived from this footage.

    A fixed default is the wrong thing here: the right cut-off depends on the
    camera, the mount and how much of the drive was spent stationary. A
    default of 3.0 discarded a fifth of perfectly good frames on one test set
    while catching none of the stationary ones on another.
    """
    want_auto = [v for v in (args.min_sharpness, args.min_difference)
                 if str(v).lower() == "auto"]
    if not want_auto:
        return float(args.min_sharpness), float(args.min_difference)

    print("Calibrating quality thresholds on a sample of the footage...")
    sharp_a, diff_a = sample_scores(videos, args.interval,
                                    args.crop_top, args.crop_side)
    resolved = []
    for value, values, label in ((args.min_sharpness, sharp_a, "sharpness"),
                                 (args.min_difference, diff_a, "difference")):
        if str(value).lower() != "auto":
            resolved.append(float(value))
            continue
        thr, split, reason = suggest_threshold(values)
        if reason:
            print(f"  {label:<11} no filtering — {reason}")
        else:
            print(f"  {label:<11} {thr:.1f}  "
                  f"(discards ~{split['drop_rate']:.0f}%, low cluster "
                  f"{split['low_mean']:.1f} vs normal {split['high_mean']:.1f})")
        resolved.append(thr)
    print()
    return resolved[0], resolved[1]


def cmd_extract(args):
    rows = read_labels(args.labels)
    if not rows:
        print(f"No usable rows in {args.labels}. Fill in the 'label' column.")
        return

    video_dir = Path(args.videos)
    all_videos = find_videos(video_dir)
    args.min_sharpness, args.min_difference = resolve_thresholds(
        args, all_videos)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    kept_by = Counter()
    sessions = defaultdict(Counter)
    dropped = Counter()
    missing = []

    for row in rows:
        path = video_dir / row["video"]
        if not path.exists():
            hits = list(video_dir.rglob(row["video"]))
            if not hits:
                missing.append(row["video"])
                continue
            path = hits[0]

        dest = out_root / row["session"] / row["label"]
        dest.mkdir(parents=True, exist_ok=True)

        span = ""
        if row["start"] is not None or row["end"] is not None:
            span = f" [{row['start'] or 0:.0f}s–{row['end'] or 0:.0f}s]"
        print(f"\n{path.name}{span}  ->  {row['session']}/{row['label']}")

        prev = None
        kept = blurry = static = 0
        existing = len(list(dest.glob("*.jpg")))

        for t, frame in iter_samples(path, args.interval,
                                     row["start"], row["end"]):
            region = road_region(frame, args.crop_top, args.crop_side)
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

            if sharpness(gray) < args.min_sharpness:
                blurry += 1
                continue

            small = cv2.resize(gray, (64, 64))
            if difference(small, prev) < args.min_difference:
                static += 1
                continue
            prev = small

            out = frame
            if args.max_width and frame.shape[1] > args.max_width:
                scale = args.max_width / frame.shape[1]
                out = cv2.resize(frame, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA)

            name = f"{row['session']}_{existing + kept:06d}_{int(t):05d}s.jpg"
            cv2.imwrite(str(dest / name), out,
                        [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            kept += 1

        print(f"  kept {kept:>5}   dropped: {blurry} blurred, {static} static")
        kept_by[row["label"]] += kept
        sessions[row["session"]][row["label"]] += kept
        dropped["blur"] += blurry
        dropped["static"] += static

    # ---- summary -------------------------------------------------------
    print(f"\n{'=' * 68}\nExtraction summary\n{'=' * 68}")
    print(f"{'session':<32}" + "".join(f"{c:>9}" for c in CLASSES))
    print("-" * 68)
    for session in sorted(sessions):
        counts = sessions[session]
        print(f"{session:<32}" + "".join(f"{counts.get(c, 0):>9}"
                                         for c in CLASSES))
    print("-" * 68)
    total = sum(kept_by.values())
    print(f"{'TOTAL':<32}" + "".join(f"{kept_by.get(c, 0):>9}" for c in CLASSES))
    print(f"\n{total} frames kept, {sum(dropped.values())} dropped "
          f"({dropped['blur']} blurred, {dropped['static']} static)")

    if missing:
        print(f"\n  ! {len(missing)} file(s) in the sheet were not found: "
              f"{', '.join(missing[:5])}")

    # ---- warnings that matter for the split ----------------------------
    print()
    n_sessions = len(sessions)
    if n_sessions < 3 * len(CLASSES):
        print(f"  ! {n_sessions} distinct sessions. The split needs at least "
              f"{3 * len(CLASSES)} to put every class in every split.")
    for c in CLASSES:
        n = sum(1 for s in sessions.values() if s.get(c, 0))
        if n == 0:
            print(f"  ! no '{c}' frames at all.")
        elif n < 3:
            print(f"  ! '{c}' appears in only {n} session(s) — it cannot be "
                  f"present in all three splits.")
    if total and kept_by.get("ice", 0) / total < 0.05:
        print(f"  ! ice is {100 * kept_by.get('ice', 0) / max(total, 1):.1f}% "
              f"of the dataset. Expect weak ice recall; keep the class "
              f"weighting on and report per-class metrics.")

    print(f"\nFrames written to {out_root}")
    print("Zip it for Colab:  zip -r frames.zip " + str(out_root))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--videos", default="./footage",
                        help="directory containing the recorded drives")
        sp.add_argument("--labels", default="labels.csv")
        sp.add_argument("--interval", type=float, default=3.0,
                        help="seconds between sampled frames (default 3)")
        sp.add_argument("--crop_top", type=float, default=0.45)
        sp.add_argument("--crop_side", type=float, default=0.15)
        return sp

    s = common(sub.add_parser("scan", help="write the labelling sheet"))
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_scan)

    r = common(sub.add_parser("report", help="inspect quality score ranges"))
    r.add_argument("--sample_videos", type=int, default=3)
    r.add_argument("--sample_frames", type=int, default=120)
    r.set_defaults(func=cmd_report)

    e = common(sub.add_parser("extract", help="write the frames"))
    e.add_argument("--out", default="./frames")
    e.add_argument("--min_sharpness", default="auto",
                   help="blur cut-off, or 'auto' to derive it from the "
                        "footage (default), or 0 to disable")
    e.add_argument("--min_difference", default="auto",
                   help="near-duplicate cut-off, or 'auto' (default), or 0")
    e.add_argument("--max_width", type=int, default=960,
                   help="downscale wider frames (0 keeps full size)")
    e.add_argument("--quality", type=int, default=92)
    e.set_defaults(func=cmd_extract)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
