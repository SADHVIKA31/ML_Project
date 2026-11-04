"""
dataset_audit.py
----------------
Phase 1 – Steps 3-8: Full dataset audit.

Covers:
  3. Understand dataset structure
  4. Count total images
  5. Count number of species/classes
  6. Count images per species
  7. Identify missing / corrupted / duplicate / unreadable images
  8. Analyse class imbalance

Usage:
    python scripts/dataset_audit.py --dataset data/processed --report reports/dataset_audit.json

Requirements:
    pip install Pillow imagehash tqdm
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_md5(path: Path, chunk: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _is_valid_image(path: Path) -> tuple[bool, Optional[str]]:
    """Return (is_valid, error_message)."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()          # basic integrity check
        return True, None
    except Exception as e:
        return False, str(e)


def _perceptual_hash(path: Path) -> Optional[str]:
    """Return perceptual hash string, or None if the image cannot be opened."""
    try:
        import imagehash
        from PIL import Image
        with Image.open(path) as img:
            return str(imagehash.phash(img))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------

def audit_dataset(dataset_dir: Path, use_phash: bool = False) -> dict:
    print(f"[INFO] Auditing dataset at: {dataset_dir.resolve()}")

    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    classes = [d.name for d in class_dirs]

    total_images = 0
    per_class: dict[str, int] = {}
    corrupted: list[dict] = []
    unreadable: list[str] = []
    duplicates: list[dict] = []

    # For duplicate detection
    md5_map: dict[str, list[str]] = defaultdict(list)
    phash_map: dict[str, list[str]] = defaultdict(list)

    try:
        from tqdm import tqdm
        progress = tqdm
    except ImportError:
        def progress(x, **kwargs): return x  # type: ignore

    for cls_dir in class_dirs:
        images = [
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        per_class[cls_dir.name] = len(images)
        total_images += len(images)

        for img_path in progress(images, desc=cls_dir.name, leave=False):
            # Corruption / readability check
            valid, err = _is_valid_image(img_path)
            if not valid:
                corrupted.append({"path": str(img_path), "error": err})

            # MD5 duplicate check
            try:
                md5 = _file_md5(img_path)
                md5_map[md5].append(str(img_path))
            except Exception as e:
                unreadable.append(str(img_path))

            # Perceptual hash (optional, slower)
            if use_phash:
                ph = _perceptual_hash(img_path)
                if ph:
                    phash_map[ph].append(str(img_path))

    # Identify duplicate groups
    for md5, paths in md5_map.items():
        if len(paths) > 1:
            duplicates.append({"md5": md5, "files": paths})

    phash_duplicates = []
    if use_phash:
        for ph, paths in phash_map.items():
            if len(paths) > 1:
                phash_duplicates.append({"phash": ph, "files": paths})

    # Class imbalance statistics
    counts = list(per_class.values())
    if counts:
        avg = sum(counts) / len(counts)
        imbalance_ratio = max(counts) / max(min(counts), 1)
    else:
        avg = 0
        imbalance_ratio = 0

    report = {
        "dataset_path": str(dataset_dir.resolve()),
        "num_classes": len(classes),
        "classes": classes,
        "total_images": total_images,
        "images_per_class": per_class,
        "statistics": {
            "mean_images_per_class": round(avg, 2),
            "max_images": max(counts) if counts else 0,
            "min_images": min(counts) if counts else 0,
            "imbalance_ratio_max_to_min": round(imbalance_ratio, 2),
        },
        "corrupted_images": corrupted,
        "unreadable_images": unreadable,
        "exact_duplicates": duplicates,
        "perceptual_duplicates": phash_duplicates if use_phash else "not computed (use --phash)",
    }

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Audit an organised image dataset")
    parser.add_argument("--dataset", default="data/processed", help="Organised dataset directory")
    parser.add_argument("--report", default="reports/dataset_audit.json", help="Output JSON report path")
    parser.add_argument("--phash", action="store_true", help="Enable perceptual hash duplicate detection (slow)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        import sys; sys.exit(f"[ERROR] Dataset directory not found: {dataset_dir}")

    report = audit_dataset(dataset_dir, use_phash=args.phash)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Pretty print summary
    print("\n" + "=" * 60)
    print("  DATASET AUDIT SUMMARY")
    print("=" * 60)
    print(f"  Total classes     : {report['num_classes']}")
    print(f"  Total images      : {report['total_images']}")
    s = report["statistics"]
    print(f"  Mean / class      : {s['mean_images_per_class']}")
    print(f"  Max  / class      : {s['max_images']}")
    print(f"  Min  / class      : {s['min_images']}")
    print(f"  Imbalance ratio   : {s['imbalance_ratio_max_to_min']}x")
    print(f"  Corrupted images  : {len(report['corrupted_images'])}")
    print(f"  Exact duplicates  : {len(report['exact_duplicates'])} groups")
    print("=" * 60)
    print(f"\n[DONE] Full report saved to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
