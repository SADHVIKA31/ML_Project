"""
verify_leakage.py
-----------------
Phase 2 – Steps 5 & 6: Verify that splits are properly separated and that no
near-duplicate image leakage exists between training and testing.

Three verification layers
--------------------------
1. VIDEO-GROUP CHECK  – Confirm no video/trajectory id appears in more than
   one split.  This is the primary leakage guard.

2. EXACT DUPLICATE CHECK – MD5 hash every image; flag any file that appears
   in both train and test (or val and test).

3. NEAR-DUPLICATE CHECK (optional, --phash) – Perceptual hash (pHash) every
   image with hamming-distance ≤ threshold to find visually similar images
   that slipped across split boundaries.

Output
------
  reports/leakage_report.json
  reports/leakage_report.md   (human-readable summary)

Usage
------
  python scripts/verify_leakage.py \\
      --splits   data/splits \\
      --dataset  data/processed \\
      --out      reports \\
      [--phash]  [--phash-threshold 8]

Requirements: pip install Pillow imagehash tqdm pandas
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _md5(path: Path, chunk: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _phash(path: Path):
    import imagehash
    from PIL import Image
    with Image.open(path) as img:
        return imagehash.phash(img)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_video_group_leakage(
    train: list[dict], val: list[dict], test: list[dict]
) -> dict:
    """Ensure no video_id appears in more than one split."""
    train_vids = {r["video_id"] for r in train}
    val_vids   = {r["video_id"] for r in val}
    test_vids  = {r["video_id"] for r in test}

    train_test_leak = train_vids & test_vids
    train_val_leak  = train_vids & val_vids
    val_test_leak   = val_vids   & test_vids

    return {
        "passed": not (train_test_leak or train_val_leak or val_test_leak),
        "train_test_shared_videos": sorted(train_test_leak),
        "train_val_shared_videos":  sorted(train_val_leak),
        "val_test_shared_videos":   sorted(val_test_leak),
    }


def check_exact_duplicates(
    train: list[dict], val: list[dict], test: list[dict]
) -> dict:
    """MD5-hash every image; find hashes shared across splits."""
    def hash_records(records: list[dict]) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = defaultdict(list)
        for r in tqdm(records, desc="  Hashing", leave=False):
            p = Path(r["image_path"])
            if p.exists():
                mapping[_md5(p)].append(r["image_path"])
        return mapping

    print("[INFO] Hashing train images ...")
    train_hashes = hash_records(train)
    print("[INFO] Hashing val images ...")
    val_hashes   = hash_records(val)
    print("[INFO] Hashing test images ...")
    test_hashes  = hash_records(test)

    train_test_dup = {h: (train_hashes[h], test_hashes[h])
                      for h in train_hashes if h in test_hashes}
    train_val_dup  = {h: (train_hashes[h], val_hashes[h])
                      for h in train_hashes if h in val_hashes}
    val_test_dup   = {h: (val_hashes[h],   test_hashes[h])
                      for h in val_hashes   if h in test_hashes}

    return {
        "passed": not (train_test_dup or train_val_dup or val_test_dup),
        "train_test_exact_duplicates": [
            {"md5": h, "train": a, "test": b}
            for h, (a, b) in train_test_dup.items()
        ],
        "train_val_exact_duplicates": [
            {"md5": h, "train": a, "val": b}
            for h, (a, b) in train_val_dup.items()
        ],
        "val_test_exact_duplicates": [
            {"md5": h, "val": a, "test": b}
            for h, (a, b) in val_test_dup.items()
        ],
    }


def check_near_duplicates(
    train: list[dict], test: list[dict], threshold: int = 8
) -> dict:
    """
    Perceptual hash comparison between train and test.
    Pairs with hamming distance ≤ threshold are flagged.
    """
    print("[INFO] Computing perceptual hashes for train ...")
    train_ph = []
    for r in tqdm(train, desc="  pHash train", leave=False):
        p = Path(r["image_path"])
        if p.exists():
            try:
                train_ph.append((_phash(p), r["image_path"]))
            except Exception:
                pass

    print("[INFO] Computing perceptual hashes for test ...")
    test_ph = []
    for r in tqdm(test, desc="  pHash test", leave=False):
        p = Path(r["image_path"])
        if p.exists():
            try:
                test_ph.append((_phash(p), r["image_path"]))
            except Exception:
                pass

    near_dup_pairs = []
    print("[INFO] Comparing hashes (this may take a while) ...")
    for tr_hash, tr_path in tqdm(train_ph, desc="  Comparing", leave=False):
        for te_hash, te_path in test_ph:
            dist = tr_hash - te_hash
            if dist <= threshold:
                near_dup_pairs.append({
                    "train_image":    tr_path,
                    "test_image":     te_path,
                    "hamming_distance": dist,
                })

    return {
        "passed":           len(near_dup_pairs) == 0,
        "threshold":        threshold,
        "near_dup_pairs":   near_dup_pairs[:100],   # cap at 100 in report
        "total_flagged":    len(near_dup_pairs),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _write_markdown(report: dict, path: Path):
    lines = [
        "# Leakage Verification Report",
        "",
        "## Summary",
        "",
        "| Check | Result |",
        "|-------|--------|",
    ]

    vg = report["video_group_check"]
    ed = report["exact_duplicate_check"]
    lines.append(f"| Video-Group Isolation | {'✅ PASS' if vg['passed'] else '❌ FAIL'} |")
    lines.append(f"| Exact Duplicate Check | {'✅ PASS' if ed['passed'] else '❌ FAIL'} |")

    if "near_duplicate_check" in report:
        nd = report["near_duplicate_check"]
        lines.append(f"| Near-Duplicate Check (pHash ≤ {nd['threshold']}) | {'✅ PASS' if nd['passed'] else '❌ FAIL'} |")

    lines += [
        "",
        "## 1. Video-Group Isolation",
        "",
    ]
    if vg["passed"]:
        lines.append("> ✅ No video/trajectory ID appears in more than one split.")
    else:
        lines.append("> ❌ **Leakage detected!** Shared video IDs found:")
        for pair, label in [
            ("train_test_shared_videos", "Train ∩ Test"),
            ("train_val_shared_videos",  "Train ∩ Val"),
            ("val_test_shared_videos",   "Val   ∩ Test"),
        ]:
            if vg[pair]:
                lines.append(f"\n**{label}:** {', '.join(vg[pair][:20])}")

    lines += ["", "## 2. Exact Duplicate Check", ""]
    if ed["passed"]:
        lines.append("> ✅ No exact duplicate images found across splits.")
    else:
        for pair, label in [
            ("train_test_exact_duplicates", "Train ↔ Test"),
            ("train_val_exact_duplicates",  "Train ↔ Val"),
            ("val_test_exact_duplicates",   "Val   ↔ Test"),
        ]:
            if ed[pair]:
                lines.append(f"\n**{label} duplicates ({len(ed[pair])}):**")
                for item in ed[pair][:10]:
                    lines.append(f"- MD5 `{item['md5']}`")

    if "near_duplicate_check" in report:
        nd = report["near_duplicate_check"]
        lines += ["", f"## 3. Near-Duplicate Check (pHash threshold ≤ {nd['threshold']})", ""]
        if nd["passed"]:
            lines.append("> ✅ No near-duplicate images found between train and test.")
        else:
            lines.append(f"> ❌ **{nd['total_flagged']} near-duplicate pair(s) detected.** First 10:")
            for pair in nd["near_dup_pairs"][:10]:
                lines.append(
                    f"- Hamming {pair['hamming_distance']}: "
                    f"`{Path(pair['train_image']).name}` ↔ `{Path(pair['test_image']).name}`"
                )

    lines += [
        "",
        "---",
        "*Auto-generated by `verify_leakage.py`*",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify leakage between splits")
    parser.add_argument("--splits",          default="data/splits",    help="Split directory")
    parser.add_argument("--out",             default="reports",        help="Output directory for reports")
    parser.add_argument("--phash",           action="store_true",      help="Run perceptual hash near-dup check")
    parser.add_argument("--phash-threshold", type=int, default=8,      help="Max hamming distance to flag (default 8)")
    args = parser.parse_args()

    splits_dir = Path(args.splits)
    for split_name in ("train", "val", "test"):
        p = splits_dir / f"{split_name}.csv"
        if not p.exists():
            import sys; sys.exit(f"[ERROR] {p} not found. Run split_dataset.py first.")

    train = _read_csv(splits_dir / "train.csv")
    val   = _read_csv(splits_dir / "val.csv")
    test  = _read_csv(splits_dir / "test.csv")

    print(f"[INFO] Loaded splits  — train: {len(train)}, val: {len(val)}, test: {len(test)}")

    report: dict = {}

    print("\n[CHECK 1] Video-group isolation ...")
    report["video_group_check"] = check_video_group_leakage(train, val, test)

    print("\n[CHECK 2] Exact duplicate detection ...")
    report["exact_duplicate_check"] = check_exact_duplicates(train, val, test)

    if args.phash:
        print(f"\n[CHECK 3] Near-duplicate detection (pHash threshold ≤ {args.phash_threshold}) ...")
        try:
            import imagehash  # noqa: F401
        except ImportError:
            import sys
            sys.exit("[ERROR] 'imagehash' not installed. pip install imagehash")
        report["near_duplicate_check"] = check_near_duplicates(train, test, threshold=args.phash_threshold)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "leakage_report.json"
    md_path   = out_dir / "leakage_report.md"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    _write_markdown(report, md_path)

    # --- Console summary ---
    print("\n" + "=" * 55)
    print("  LEAKAGE VERIFICATION RESULTS")
    print("=" * 55)
    vg = report["video_group_check"]
    ed = report["exact_duplicate_check"]
    print(f"  Video-Group Isolation : {'✅ PASS' if vg['passed'] else '❌ FAIL'}")
    print(f"  Exact Duplicates      : {'✅ PASS' if ed['passed'] else '❌ FAIL'}")
    if "near_duplicate_check" in report:
        nd = report["near_duplicate_check"]
        print(f"  Near-Duplicates       : {'✅ PASS' if nd['passed'] else f'❌ FAIL ({nd[\"total_flagged\"]} pairs)'}")
    print("=" * 55)
    print(f"\n[DONE] Reports saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
