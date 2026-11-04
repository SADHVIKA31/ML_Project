"""
organise_dataset.py
-------------------
Phase 1 – Step 2 & 3: Organise raw dataset into a clean folder structure.

Expected output layout (data/processed/):
    data/
    └── processed/
        └── <ClassName>/
            ├── img_001.jpg
            ├── img_002.jpg
            └── ...

Usage:
    python scripts/organise_dataset.py --src data/raw --dest data/processed

The script handles:
  - Flat image directories
  - Already class-separated subdirectories
  - Mixed image formats (jpg, jpeg, png, bmp, tiff)
"""

import argparse
import shutil
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def organise(src: Path, dest: Path) -> dict:
    """Walk src, group images by parent-folder name, copy to dest."""
    dest.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}

    all_images = [
        p for p in src.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not all_images:
        print(f"[WARN] No image files found under {src}")
        return stats

    # Determine if images are already in class sub-folders
    # (i.e., the immediate parent of each image is not src itself)
    has_class_folders = any(img.parent != src for img in all_images)

    for img in all_images:
        class_name = img.parent.name if has_class_folders else "unknown"
        target_dir = dest / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        target_file = target_dir / img.name
        # Avoid overwriting duplicates by appending a counter
        if target_file.exists():
            stem, suffix = img.stem, img.suffix
            counter = 1
            while target_file.exists():
                target_file = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        shutil.copy2(img, target_file)
        stats[class_name] = stats.get(class_name, 0) + 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Organise dataset into class folders")
    parser.add_argument("--src", default="data/raw", help="Raw dataset directory")
    parser.add_argument("--dest", default="data/processed", help="Organised output directory")
    args = parser.parse_args()

    src = Path(args.src)
    dest = Path(args.dest)

    if not src.exists():
        import sys; sys.exit(f"[ERROR] Source directory not found: {src}")

    print(f"[INFO] Organising '{src}' -> '{dest}' ...")
    stats = organise(src, dest)

    print("\n[DONE] Organisation complete.")
    print(f"{'Class':<40} {'Images':>8}")
    print("-" * 50)
    for cls, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"{cls:<40} {count:>8}")
    print("-" * 50)
    print(f"{'TOTAL':<40} {sum(stats.values()):>8}")


if __name__ == "__main__":
    main()
