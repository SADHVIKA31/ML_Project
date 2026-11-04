"""
extract_video_groups.py
-----------------------
Phase 2 – Steps 1 & 2: Identify source videos / trajectories associated with
each image and group them accordingly.

The Fish4Knowledge dataset encodes the source video in the filename:
    fish_image_<video_id>_<frame_id>.png
    e.g.  fish_image_00001_0001.png  → video_id = 00001

For datasets whose filenames carry no such structure, EXIF metadata is used
as a fallback.

Output:
    data/interim/video_groups.json
    {
      "video_id_1": ["path/img1.jpg", "path/img2.jpg", ...],
      "video_id_2": [...],
      ...
    }

Usage:
    python scripts/extract_video_groups.py --dataset data/processed --out data/interim/video_groups.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ---------------------------------------------------------------------------
# Filename-based grouping (primary strategy)
# ---------------------------------------------------------------------------

# Common Fish4Knowledge pattern: fish_image_VVVVV_FFFF.*
_F4K_PATTERN = re.compile(r"fish_image[_-](\d+)[_-]\d+", re.IGNORECASE)

# Generic: any run of digits separated by underscore can encode a video id.
# We use the FIRST numeric segment as the trajectory/video id.
_GENERIC_PATTERN = re.compile(r"^[^\d]*(\d+)[_-](\d+)")


def _video_id_from_filename(name: str) -> str:
    """Return a video/trajectory ID extracted from an image filename."""
    stem = Path(name).stem

    # Fish4Knowledge exact pattern
    m = _F4K_PATTERN.match(stem)
    if m:
        return f"vid_{m.group(1)}"

    # Generic two-segment numbering: first segment = video id
    m = _GENERIC_PATTERN.match(stem)
    if m:
        return f"vid_{m.group(1)}"

    # Cannot determine – treat the full filename (minus extension) as its own group
    return f"single_{stem}"


# ---------------------------------------------------------------------------
# EXIF-based grouping (fallback)
# ---------------------------------------------------------------------------

def _video_id_from_exif(path: Path) -> str | None:
    """Try to extract a grouping key from EXIF ImageDescription or UserComment."""
    if not PIL_AVAILABLE:
        return None
    try:
        with Image.open(path) as img:
            exif_data = img._getexif()  # type: ignore[attr-defined]
            if not exif_data:
                return None
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, str(tag_id))
                if tag in ("ImageDescription", "UserComment"):
                    m = re.search(r"vid(?:eo)?[_-]?(\d+)", str(value), re.IGNORECASE)
                    if m:
                        return f"vid_{m.group(1)}"
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main grouping logic
# ---------------------------------------------------------------------------

def build_video_groups(dataset_dir: Path) -> dict[str, list[str]]:
    """Return {video_id: [relative_image_paths]} grouped by source video."""
    groups: dict[str, list[str]] = defaultdict(list)
    all_images = [
        p for p in dataset_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not all_images:
        print(f"[WARN] No images found under {dataset_dir}")
        return {}

    print(f"[INFO] Processing {len(all_images)} images ...")

    for img_path in all_images:
        vid_id = _video_id_from_filename(img_path.name)

        # If filename gave only a "single_" fallback, try EXIF
        if vid_id.startswith("single_"):
            exif_id = _video_id_from_exif(img_path)
            if exif_id:
                vid_id = exif_id

        groups[vid_id].append(str(img_path.relative_to(dataset_dir)))

    # Sort paths within each group for reproducibility
    for vid_id in groups:
        groups[vid_id].sort()

    print(f"[INFO] Found {len(groups)} unique video/trajectory groups.")
    return dict(groups)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Group images by source video/trajectory"
    )
    parser.add_argument("--dataset", default="data/processed", help="Organised dataset dir")
    parser.add_argument("--out", default="data/interim/video_groups.json", help="Output JSON path")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        sys.exit(f"[ERROR] Dataset directory not found: {dataset_dir}")

    groups = build_video_groups(dataset_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(groups, f, indent=2)

    print(f"[DONE] Video groups saved to: {out_path.resolve()}")
    print(f"\n  Groups:  {len(groups)}")
    total_imgs = sum(len(v) for v in groups.values())
    print(f"  Images:  {total_imgs}")
    singletons = sum(1 for v in groups.values() if len(v) == 1)
    print(f"  Singleton groups (no trajectory info): {singletons}")


if __name__ == "__main__":
    main()
