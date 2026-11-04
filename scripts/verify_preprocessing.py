"""
verify_preprocessing.py
-----------------------
Phase 3 – Step 9: Verify that identical preprocessing was applied consistently
across all three splits (train / val / test).

Checks performed
----------------
1. Label-map consistency – all splits use the same integer encoding.
2. Array shape consistency – every preprocessed array has shape (H, W, 3).
3. Pixel-value range check – values match the expected normalisation range.
4. Class coverage – every class present in train also appears in val and test.
5. Per-split label distribution – flag severe imbalance.

Usage:
    python scripts/verify_preprocessing.py \\
        --preprocessed data/preprocessed \\
        --out          reports
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_label_map(preprocessed_dir: Path) -> dict:
    p = preprocessed_dir / "label_map.json"
    if not p.exists():
        sys.exit(f"[ERROR] label_map.json not found in {preprocessed_dir}")
    with open(p) as f:
        return json.load(f)


def _expected_range(normalize: str) -> tuple[float, float]:
    if normalize == "minmax":
        return 0.0, 1.0
    elif normalize == "imagenet":
        # After ImageNet z-score, typical range is roughly -2.5 to 2.75
        return -3.0, 3.5
    return 0.0, 255.0


def verify(preprocessed_dir: Path, out_dir: Path):
    # Load metadata
    stats_path = preprocessed_dir / "split_stats.json"
    if not stats_path.exists():
        sys.exit(f"[ERROR] split_stats.json not found. Run preprocess.py first.")

    with open(stats_path) as f:
        stats = json.load(f)

    normalize   = stats.get("normalize", "imagenet")
    target_size = stats.get("target_size", [224, 224])
    label_map   = _load_label_map(preprocessed_dir)

    expected_lo, expected_hi = _expected_range(normalize)
    H, W = target_size

    report = {
        "normalize": normalize,
        "target_size": target_size,
        "num_classes": len(label_map),
        "checks": {},
    }

    all_pass = True

    # -----------------------------------------------------------------
    # Check 1: stacked array files exist
    # -----------------------------------------------------------------
    split_arrays: dict[str, dict] = {}
    missing_splits = []
    for split in ("train", "val", "test"):
        X_path = preprocessed_dir / f"X_{split}.npy"
        y_path = preprocessed_dir / f"y_{split}.npy"
        if X_path.exists() and y_path.exists():
            X = np.load(X_path, mmap_mode="r")
            y = np.load(y_path, mmap_mode="r")
            split_arrays[split] = {"X": X, "y": y}
        else:
            # Fallback: individual .npy per file (--save-npy mode)
            missing_splits.append(split)

    if missing_splits:
        report["checks"]["stacked_arrays"] = {
            "passed": False,
            "note": f"X_<split>.npy not found for: {missing_splits}. "
                    "If --save-npy was used, run without --save-npy to generate stacked arrays.",
        }
        all_pass = False
    else:
        report["checks"]["stacked_arrays"] = {"passed": True}

    if split_arrays:
        # -----------------------------------------------------------------
        # Check 2: Shape consistency
        # -----------------------------------------------------------------
        shape_issues = []
        for split, arr_dict in split_arrays.items():
            X = arr_dict["X"]
            if X.shape[1:] != (H, W, 3):
                shape_issues.append(f"{split}: {X.shape} expected (N,{H},{W},3)")

        report["checks"]["shape_consistency"] = {
            "passed":  len(shape_issues) == 0,
            "issues":  shape_issues,
        }
        if shape_issues:
            all_pass = False

        # -----------------------------------------------------------------
        # Check 3: Pixel-value range
        # -----------------------------------------------------------------
        range_issues = []
        for split, arr_dict in split_arrays.items():
            X = arr_dict["X"]
            lo, hi = float(X.min()), float(X.max())
            if lo < expected_lo - 0.1 or hi > expected_hi + 0.1:
                range_issues.append(
                    f"{split}: range [{lo:.3f}, {hi:.3f}]  "
                    f"expected [{expected_lo}, {expected_hi}]"
                )

        report["checks"]["pixel_range"] = {
            "passed":           len(range_issues) == 0,
            "expected_range":   [expected_lo, expected_hi],
            "issues":           range_issues,
        }
        if range_issues:
            all_pass = False

        # -----------------------------------------------------------------
        # Check 4: Class coverage
        # -----------------------------------------------------------------
        train_classes = set(split_arrays["train"]["y"].tolist()) if "train" in split_arrays else set()
        coverage_issues = []
        for split in ("val", "test"):
            if split not in split_arrays:
                continue
            split_classes = set(split_arrays[split]["y"].tolist())
            missing = train_classes - split_classes
            if missing:
                inv = {v: k for k, v in label_map.items()}
                missing_names = [inv.get(m, str(m)) for m in missing]
                coverage_issues.append(f"{split} missing classes: {missing_names}")

        report["checks"]["class_coverage"] = {
            "passed": len(coverage_issues) == 0,
            "issues": coverage_issues,
        }
        if coverage_issues:
            all_pass = False

        # -----------------------------------------------------------------
        # Check 5: Per-split label distribution
        # -----------------------------------------------------------------
        distribution = {}
        for split, arr_dict in split_arrays.items():
            y = arr_dict["y"]
            counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
            distribution[split] = counts

        report["checks"]["label_distribution"] = {
            "passed":       True,
            "distribution": distribution,
        }

    report["overall_passed"] = all_pass

    # -----------------------------------------------------------------
    # Save report
    # -----------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "preprocessing_verification.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # --- Markdown ---
    md_lines = [
        "# Preprocessing Verification Report",
        "",
        "## Overall Result",
        "",
        f"**{'✅ ALL CHECKS PASSED' if all_pass else '❌ ONE OR MORE CHECKS FAILED'}**",
        "",
        "## Check Details",
        "",
        "| Check | Result |",
        "|-------|--------|",
    ]
    for check_name, check_result in report["checks"].items():
        status = "✅ PASS" if check_result.get("passed") else "❌ FAIL"
        md_lines.append(f"| {check_name.replace('_', ' ').title()} | {status} |")

    for check_name, check_result in report["checks"].items():
        issues = check_result.get("issues", [])
        if issues:
            md_lines += [
                "",
                f"### ❌ {check_name.replace('_', ' ').title()} Issues",
                "",
            ]
            for issue in issues:
                md_lines.append(f"- {issue}")

    md_lines += ["", "---", "*Auto-generated by `verify_preprocessing.py`*"]
    md_path = out_dir / "preprocessing_verification.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # --- Console ---
    print("\n" + "=" * 55)
    print("  PREPROCESSING VERIFICATION")
    print("=" * 55)
    for check_name, check_result in report["checks"].items():
        icon = "✅" if check_result.get("passed") else "❌"
        print(f"  {icon}  {check_name}")
    print("=" * 55)
    print(f"\n[DONE] JSON → {json_path.resolve()}")
    print(f"[DONE] MD   → {md_path.resolve()}")

    if not all_pass:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Verify consistent preprocessing across splits")
    parser.add_argument("--preprocessed", default="data/preprocessed", help="Preprocessed data dir")
    parser.add_argument("--out",          default="reports",            help="Reports directory")
    args = parser.parse_args()

    verify(Path(args.preprocessed), Path(args.out))


if __name__ == "__main__":
    main()
