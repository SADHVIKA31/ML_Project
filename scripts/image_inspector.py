"""
image_inspector.py
------------------
Phase 3 – Steps 1 & 2: Load images with OpenCV/Pillow and audit their
dimensions, formats, channels and colour modes.

Produces:
  data/interim/image_inspection.csv   – per-image audit table
  data/interim/image_inspection.json  – summary statistics

Usage:
    python scripts/image_inspector.py --dataset data/processed --out data/interim
"""

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def inspect_image(path: Path) -> dict:
    """Return per-image metadata dict."""
    record = {
        "path":        str(path),
        "filename":    path.name,
        "class":       path.parent.name,
        "extension":   path.suffix.lower(),
        "size_bytes":  path.stat().st_size,
        "width":       None,
        "height":      None,
        "channels":    None,
        "mode":        None,          # PIL mode: RGB, L, RGBA …
        "readable":    True,
        "error":       None,
    }

    # --- Pillow read ---
    try:
        with Image.open(path) as img:
            record["width"]    = img.width
            record["height"]   = img.height
            record["mode"]     = img.mode
            record["channels"] = len(img.getbands())
    except Exception as e:
        record["readable"] = False
        record["error"]    = str(e)
        return record

    # --- OpenCV read (optional cross-check) ---
    if CV2_AVAILABLE:
        try:
            arr = cv2.imread(str(path))
            if arr is None:
                record["readable"] = False
                record["error"]    = "cv2.imread returned None"
            else:
                # Overwrite with OpenCV values (ground truth for model input)
                record["height"], record["width"] = arr.shape[:2]
                record["channels"] = arr.shape[2] if arr.ndim == 3 else 1
        except Exception as e:
            record["error"] = f"cv2 error: {e}"

    return record


def run_inspection(dataset_dir: Path, out_dir: Path):
    all_images = sorted([
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not all_images:
        sys.exit(f"[ERROR] No images found under {dataset_dir}")

    print(f"[INFO] Inspecting {len(all_images)} images ...")

    try:
        from tqdm import tqdm
        iterable = tqdm(all_images, desc="Inspecting")
    except ImportError:
        iterable = all_images

    records = [inspect_image(p) for p in iterable]

    # --- CSV ---
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "image_inspection.csv"
    fieldnames = ["path", "filename", "class", "extension", "size_bytes",
                  "width", "height", "channels", "mode", "readable", "error"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # --- Summary JSON ---
    readable   = [r for r in records if r["readable"]]
    unreadable = [r for r in records if not r["readable"]]

    widths    = [r["width"]    for r in readable if r["width"]]
    heights   = [r["height"]   for r in readable if r["height"]]
    channels  = [r["channels"] for r in readable if r["channels"]]
    modes     = {}
    for r in readable:
        modes[r["mode"]] = modes.get(r["mode"], 0) + 1
    extensions = {}
    for r in records:
        extensions[r["extension"]] = extensions.get(r["extension"], 0) + 1

    def _uniq_dims(readable):
        dims = set()
        for r in readable:
            if r["width"] and r["height"]:
                dims.add((r["width"], r["height"]))
        return [list(d) for d in sorted(dims)]

    summary = {
        "total_images":    len(records),
        "readable":        len(readable),
        "unreadable":      len(unreadable),
        "unreadable_paths": [r["path"] for r in unreadable],
        "unique_dimensions": _uniq_dims(readable),
        "num_unique_dimensions": len(_uniq_dims(readable)),
        "width_range":    [min(widths),  max(widths)]  if widths  else [],
        "height_range":   [min(heights), max(heights)] if heights else [],
        "channel_counts": sorted(set(channels)),
        "colour_modes":   modes,
        "extensions":     extensions,
    }

    json_path = out_dir / "image_inspection.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 55)
    print("  IMAGE INSPECTION SUMMARY")
    print("=" * 55)
    print(f"  Total images      : {summary['total_images']}")
    print(f"  Readable          : {summary['readable']}")
    print(f"  Unreadable        : {summary['unreadable']}")
    print(f"  Unique dimensions : {summary['num_unique_dimensions']}")
    print(f"  Width range       : {summary['width_range']}")
    print(f"  Height range      : {summary['height_range']}")
    print(f"  Channels found    : {summary['channel_counts']}")
    print(f"  Colour modes      : {summary['colour_modes']}")
    print("=" * 55)
    print(f"\n[DONE] CSV  → {csv_path.resolve()}")
    print(f"[DONE] JSON → {json_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Inspect image dimensions and formats")
    parser.add_argument("--dataset", default="data/processed", help="Organised dataset dir")
    parser.add_argument("--out",     default="data/interim",   help="Output directory")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        sys.exit(f"[ERROR] Dataset directory not found: {dataset_dir}")

    run_inspection(dataset_dir, Path(args.out))


if __name__ == "__main__":
    main()
