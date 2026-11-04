"""
split_dataset.py
----------------
Phase 2 – Steps 3-6: Leakage-safe dataset splitting.

Strategy
--------
1. Load video/trajectory groups produced by extract_video_groups.py.
2. Per class, group images by their video/trajectory id.
3. Shuffle video groups (not individual images) with a fixed seed.
4. Assign whole video groups to train / val / test – never splitting a group
   across two different splits (this is the core leakage-prevention guarantee).
5. Write split manifests (CSV) and optionally copy images to split folders.

Default ratios: 70% train / 15% val / 15% test  (configurable).

Output
------
  data/splits/
    train.csv
    val.csv
    test.csv
    split_summary.json

Usage
-----
  python scripts/split_dataset.py \\
      --dataset  data/processed \\
      --groups   data/interim/video_groups.json \\
      --out      data/splits \\
      --train    0.70 --val 0.15 --test 0.15 \\
      --seed     42 \\
      [--copy-files]   # physically copy images into split sub-dirs

Requirements: pip install scikit-learn pandas tqdm Pillow
"""

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_groups(groups_path: Path, dataset_dir: Path) -> dict[str, dict[str, list[Path]]]:
    """
    Returns:
        {class_name: {video_id: [absolute_image_paths]}}
    """
    with open(groups_path) as f:
        raw: dict[str, list[str]] = json.load(f)

    # raw[video_id] = [relative paths from dataset root]
    # We need to infer class from path: processed/<class>/<img>
    class_groups: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))

    for vid_id, rel_paths in raw.items():
        for rel in rel_paths:
            abs_path = dataset_dir / rel
            # Expect structure: <class_name>/<filename>
            class_name = Path(rel).parts[0] if len(Path(rel).parts) > 1 else "unknown"
            class_groups[class_name][vid_id].append(abs_path)

    return class_groups


def _split_groups(
    video_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Shuffle video_ids and split into (train, val, test) lists."""
    rng = random.Random(seed)
    ids = list(video_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = max(1, round(n * train_ratio))
    n_val   = max(1, round(n * val_ratio))
    # test gets the remainder
    train = ids[:n_train]
    val   = ids[n_train : n_train + n_val]
    test  = ids[n_train + n_val :]

    # Edge cases: if test or val ended up empty, borrow from train
    if not test and len(train) > 1:
        test = [train.pop()]
    if not val and len(train) > 1:
        val = [train.pop()]

    return train, val, test


# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------

def build_splits(
    dataset_dir: Path,
    groups_path: Path,
    out_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    copy_files: bool,
) -> dict:

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    class_groups = _load_groups(groups_path, dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_records: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    summary: dict[str, dict] = {}

    for class_name in sorted(class_groups.keys()):
        vid_map = class_groups[class_name]        # {vid_id: [paths]}
        video_ids = list(vid_map.keys())

        train_vids, val_vids, test_vids = _split_groups(
            video_ids, train_ratio, val_ratio, seed
        )

        for split_name, vids in [("train", train_vids), ("val", val_vids), ("test", test_vids)]:
            for vid_id in vids:
                for img_path in vid_map[vid_id]:
                    split_records[split_name].append({
                        "class": class_name,
                        "video_id": vid_id,
                        "image_path": str(img_path),
                        "filename": img_path.name,
                    })

        summary[class_name] = {
            "total_videos": len(video_ids),
            "train_videos": len(train_vids),
            "val_videos":   len(val_vids),
            "test_videos":  len(test_vids),
            "train_images": sum(len(vid_map[v]) for v in train_vids),
            "val_images":   sum(len(vid_map[v]) for v in val_vids),
            "test_images":  sum(len(vid_map[v]) for v in test_vids),
        }

    # --- Write CSVs ---
    fieldnames = ["class", "video_id", "image_path", "filename"]
    for split_name in ("train", "val", "test"):
        csv_path = out_dir / f"{split_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_records[split_name])
        print(f"[INFO] {split_name:5s}: {len(split_records[split_name]):5d} images → {csv_path}")

    # --- Optionally copy files ---
    if copy_files:
        print("[INFO] Copying image files into split directories ...")
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, **kw): return x  # type: ignore

        for split_name in ("train", "val", "test"):
            for record in tqdm(split_records[split_name], desc=split_name):
                src = Path(record["image_path"])
                dest = out_dir / split_name / record["class"] / record["filename"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)

    # --- Write summary JSON ---
    total_train = sum(len(v) for v in [split_records["train"]])
    total_val   = sum(len(v) for v in [split_records["val"]])
    total_test  = sum(len(v) for v in [split_records["test"]])
    total       = total_train + total_val + total_test

    full_summary = {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "totals": {
            "train": total_train,
            "val":   total_val,
            "test":  total_test,
            "total": total,
        },
        "actual_ratios": {
            "train": round(total_train / total, 4) if total else 0,
            "val":   round(total_val   / total, 4) if total else 0,
            "test":  round(total_test  / total, 4) if total else 0,
        },
        "per_class": summary,
    }

    summary_path = out_dir / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)

    return full_summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Leakage-safe dataset splitting")
    parser.add_argument("--dataset",    default="data/processed",              help="Organised dataset dir")
    parser.add_argument("--groups",     default="data/interim/video_groups.json", help="Video groups JSON")
    parser.add_argument("--out",        default="data/splits",                 help="Output directory for splits")
    parser.add_argument("--train",      type=float, default=0.70,              help="Train ratio (default 0.70)")
    parser.add_argument("--val",        type=float, default=0.15,              help="Val   ratio (default 0.15)")
    parser.add_argument("--test",       type=float, default=0.15,              help="Test  ratio (default 0.15)")
    parser.add_argument("--seed",       type=int,   default=42,                help="Random seed")
    parser.add_argument("--copy-files", action="store_true",                   help="Physically copy images into split dirs")
    args = parser.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-4:
        sys.exit("[ERROR] --train + --val + --test must equal 1.0")

    dataset_dir = Path(args.dataset)
    groups_path = Path(args.groups)

    if not dataset_dir.exists():
        sys.exit(f"[ERROR] Dataset directory not found: {dataset_dir}")
    if not groups_path.exists():
        sys.exit(f"[ERROR] Video groups file not found: {groups_path}\nRun extract_video_groups.py first.")

    summary = build_splits(
        dataset_dir  = dataset_dir,
        groups_path  = groups_path,
        out_dir      = Path(args.out),
        train_ratio  = args.train,
        val_ratio    = args.val,
        test_ratio   = args.test,
        seed         = args.seed,
        copy_files   = args.copy_files,
    )

    print("\n" + "=" * 55)
    print("  SPLIT SUMMARY")
    print("=" * 55)
    t = summary["totals"]
    r = summary["actual_ratios"]
    print(f"  Train : {t['train']:6d} images  ({r['train']*100:.1f}%)")
    print(f"  Val   : {t['val']:6d} images  ({r['val']*100:.1f}%)")
    print(f"  Test  : {t['test']:6d} images  ({r['test']*100:.1f}%)")
    print(f"  Total : {t['total']:6d} images")
    print("=" * 55)
    print(f"\n[DONE] Splits saved to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
