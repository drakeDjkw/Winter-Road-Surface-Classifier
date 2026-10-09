"""
Roboflow classification export  ->  session-grouped frames.

Roboflow splits train/valid/test by picking images at random. For footage that
is fatal: frames from one drive are near-identical, so a random split puts
near-copies of the same road on both sides and the test score measures
memorisation. The fix is to let Roboflow do the labelling, put every image in
one split, then regroup by recording session here and let the training
pipeline do the splitting.

Roboflow keeps the uploaded filename as a prefix on export
(drive_03_000123_00032s.jpg -> drive_03_000123_00032s_jpg.rf.<hash>.jpg), so
the session survives and can be read back off the name.

Usage:
    # 1. See which sessions are detected — writes nothing.
    python roboflow_to_sessions.py --export ./roboflow-export

    # 2. If the table looks right, write the regrouped copy.
    python roboflow_to_sessions.py --export ./roboflow-export \\
           --out ./frames --write
"""

import argparse
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["dry", "wet", "snow", "ice"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Matches the extractor's naming: <session>_<6-digit index>_<seconds>s
DEFAULT_PATTERN = r"^(?P<session>.+?)_\d{6}_\d+s"


def strip_roboflow_suffix(name):
    """drive_03_000123_00032s_jpg.rf.abc123.jpg -> drive_03_000123_00032s"""
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", Path(name).stem, flags=re.I)
    return re.sub(r"_(jpg|jpeg|png|bmp)$", "", stem, flags=re.I)


def session_of(filename, pattern, parts):
    clean = strip_roboflow_suffix(filename)
    if parts:
        tokens = clean.split("_")
        return "_".join(tokens[:parts]) if len(tokens) >= parts else None
    m = re.match(pattern, clean)
    return m.group("session") if m else None


def detect_format(export_dir):
    """Classification exports use class-named folders; detection exports use
    images/ + labels/ alongside a data.yaml."""
    export_dir = Path(export_dir)
    if list(export_dir.rglob("data.yaml")) or list(export_dir.rglob("labels")):
        return "detection"
    return "classification"


def read_class_names(export_dir):
    """Pull `names:` out of data.yaml without requiring PyYAML.

    Roboflow writes either a flow list (names: ['dry', 'wet']) or a block
    list over several lines, so both are handled.
    """
    yamls = list(Path(export_dir).rglob("data.yaml"))
    if not yamls:
        return None
    text = yamls[0].read_text()

    flow = re.search(r"^names:\s*\[(.*?)\]", text, re.M | re.S)
    if flow:
        return [n.strip().strip("'\"")
                for n in flow.group(1).split(",") if n.strip()]

    block = re.search(r"^names:\s*$(.*?)(?=^\S|\Z)", text, re.M | re.S)
    if block:
        names = re.findall(r"^\s*-\s*(.+?)\s*$", block.group(1), re.M)
        if names:
            return [n.strip().strip("'\"") for n in names]

    indexed = re.findall(r"^\s*(\d+)\s*:\s*(.+?)\s*$", text, re.M)
    if indexed:
        return [n.strip().strip("'\"")
                for _, n in sorted(indexed, key=lambda kv: int(kv[0]))]
    return None


def label_from_boxes(label_path, names):
    """Turn an image's box annotations into one whole-frame class.

    Road condition covers the entire frame, so every box in an image should
    carry the same class. Where they disagree the most frequent one wins and
    the image is flagged.
    """
    if not label_path.exists():
        return None, "no annotation file"
    ids = []
    for line in label_path.read_text().splitlines():
        parts_ = line.split()
        if parts_:
            try:
                ids.append(int(float(parts_[0])))
            except ValueError:
                continue
    if not ids:
        return None, "empty annotation (background image)"

    counts = Counter(ids)
    ranked = counts.most_common()
    top, n_top = ranked[0]
    if top >= len(names):
        return None, f"class id {top} outside data.yaml names"

    note = None
    if len(ranked) > 1:
        if ranked[1][1] == n_top:
            # No majority. Counter would hand back whichever class happened to
            # be listed first, which is file ordering, not evidence.
            tied = sorted(names[i] for i, c in ranked if c == n_top
                          and i < len(names))
            note = (f"TIE between {tied} ({n_top} box each) — guessed "
                    f"'{names[top]}', check this one by hand")
        else:
            note = (f"mixed classes, took the majority '{names[top]}' "
                    f"({n_top}/{len(ids)} boxes)")
    return names[top].lower(), note


def collect(export_dir, pattern, parts):
    """Walk the export, pairing every image with its class and session."""
    export_dir = Path(export_dir)
    if not export_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {export_dir}")

    fmt = detect_format(export_dir)
    names = read_class_names(export_dir) if fmt == "detection" else None
    if fmt == "detection":
        print(f"Detected an object-detection export.")
        if not names:
            raise SystemExit(
                "Could not read class names from data.yaml. Check that the "
                "export\nincludes it, or re-export in a YOLO format.")
        print(f"Classes in data.yaml: {names}")
        print("Each image's boxes are collapsed into one whole-frame label.\n")
    else:
        print("Detected a classification export (class-named folders).\n")

    found, unmatched, unknown_classes = [], [], Counter()
    skipped = Counter()
    ambiguous = []

    for img in sorted(export_dir.rglob("*")):
        if img.suffix.lower() not in IMG_EXTS or not img.is_file():
            continue

        if fmt == "detection":
            if img.parent.name.lower() != "images":
                continue
            label_path = img.parent.parent / "labels" / (img.stem + ".txt")
            label, note = label_from_boxes(label_path, names)
            if label is None:
                skipped[note] += 1
                continue
            if note:
                ambiguous.append(f"{img.name}: {note}")
        else:
            label = img.parent.name.lower()

        if label not in CLASSES:
            unknown_classes[label] += 1
            continue

        session = session_of(img.name, pattern, parts)
        if session is None:
            unmatched.append(img.name)
        else:
            found.append((img, label, session))

    for reason, n in skipped.most_common():
        print(f"  ! {n} image(s) skipped — {reason}")
    if ambiguous:
        print(f"  ! {len(ambiguous)} image(s) had mixed box classes, e.g.:")
        for a in ambiguous[:3]:
            print(f"      {a}")

    return found, unmatched, unknown_classes


def summarise(found, unmatched, unknown_classes):
    by_session = defaultdict(Counter)
    for _, label, session in found:
        by_session[session][label] += 1

    print(f"\n{'session':<34}" + "".join(f"{c:>8}" for c in CLASSES) + f"{'total':>8}")
    print("-" * 74)
    for session in sorted(by_session):
        counts = by_session[session]
        print(f"{session:<34}"
              + "".join(f"{counts.get(c, 0):>8}" for c in CLASSES)
              + f"{sum(counts.values()):>8}")
    print("-" * 74)
    totals = Counter()
    for counts in by_session.values():
        totals.update(counts)
    print(f"{'TOTAL':<34}" + "".join(f"{totals.get(c, 0):>8}" for c in CLASSES)
          + f"{sum(totals.values()):>8}")

    print(f"\n{len(by_session)} sessions, {len(found)} images")

    if unknown_classes:
        print(f"\n  ! folders that are not one of {CLASSES}, skipped: "
              + ", ".join(f"{k} ({v} images)"
                          for k, v in unknown_classes.most_common(6)))
    if unmatched:
        print(f"\n  ! {len(unmatched)} filenames did not match the session "
              f"pattern, e.g.:")
        for n in unmatched[:4]:
            print(f"      {n}")
        print("    Adjust --session_regex, or use --session_parts N to take "
              "the\n    first N underscore-separated tokens as the session "
              "name.")

    # ---- the checks that decide whether a split is even possible ------
    print()
    n_sessions = len(by_session)
    problems = False
    if n_sessions < 3 * len(CLASSES):
        print(f"  ! {n_sessions} sessions. A split that puts every class in "
              f"every part needs about {3 * len(CLASSES)}.")
        problems = True
    for c in CLASSES:
        n = sum(1 for counts in by_session.values() if counts.get(c, 0))
        if n == 0:
            print(f"  ! no '{c}' images at all.")
            problems = True
        elif n < 3:
            print(f"  ! '{c}' appears in only {n} session(s) — it cannot be "
                  f"in all three splits.")
            problems = True
    if n_sessions == 1:
        print("  ! every image resolved to a single session. The pattern is "
              "probably\n    wrong — check the filenames above.")
        problems = True
    if not problems:
        print("  Looks workable: enough sessions, every class spread across "
              "several.")
    return by_session


def write_out(found, out_dir, move=False):
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} exists and is not empty — "
                         f"remove it or choose another --out.")

    for img, label, session in found:
        dest = out_dir / session / label
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / img.name
        if move:
            shutil.move(str(img), target)
        else:
            shutil.copy2(img, target)

    print(f"\n{'moved' if move else 'copied'} {len(found)} images -> {out_dir}")
    print(f"Zip it for Colab:  zip -r frames.zip {out_dir}")
    print("Then check the split before training:")
    print(f"  python train.py --dry_run --data_root {out_dir}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", required=True,
                   help="the unzipped Roboflow export directory")
    p.add_argument("--out", default="./frames")
    p.add_argument("--session_regex", default=DEFAULT_PATTERN,
                   help="regex with a named 'session' group, matched against "
                        "the filename once Roboflow's suffix is stripped")
    p.add_argument("--session_parts", type=int, default=0,
                   help="instead of the regex, use the first N "
                        "underscore-separated tokens as the session name")
    p.add_argument("--write", action="store_true",
                   help="actually write the output (otherwise report only)")
    p.add_argument("--move", action="store_true",
                   help="move instead of copy (saves disk, destroys the export)")
    args = p.parse_args()

    found, unmatched, unknown = collect(args.export, args.session_regex,
                                        args.session_parts)

    if not found and unmatched:
        # Images are there; it is the session pattern that failed. Saying
        # "no images found" here would send them looking in the wrong place.
        print(f"Found {len(unmatched)} labelled images, but none of their "
              f"filenames matched the session pattern.\n")
        print("Examples:")
        for n in unmatched[:5]:
            print(f"    {n}")
        raise SystemExit(
            f"\nThe default pattern expects the extractor's naming,\n"
            f"  <session>_<000123>_<00032s>.jpg\n\n"
            f"For other names, give the session another way:\n"
            f"  --session_parts 2       first two underscore-separated "
            f"tokens\n"
            f"  --session_regex '...'   a regex with a named 'session' group\n\n"
            f"Whatever you choose, every image from one drive must come out "
            f"with the\nsame session name, or the split will leak.")

    if not found:
        raise SystemExit(
            f"No labelled images found under {args.export}.\n"
            f"Expected class folders named {CLASSES} (Roboflow's "
            f"classification export puts them under train/, valid/, test/).")

    summarise(found, unmatched, unknown)

    if not args.write:
        print("\nReport only — nothing written. Re-run with --write once the "
              "sessions above look right.")
        return

    write_out(found, args.out, move=args.move)


if __name__ == "__main__":
    main()
