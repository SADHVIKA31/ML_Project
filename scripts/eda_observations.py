"""
eda_observations.py
-------------------
Phase 4 – Task 11 & Deliverable: Compile EDA observations and conclusions.

Reads all previously generated EDA JSON reports and produces a final
consolidated EDA observations report:
  reports/eda/eda_observations.md

Usage
-----
  python scripts/eda_observations.py --eda-dir reports/eda
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _risk_badge(value: float, low: float, high: float,
                reverse: bool = False) -> str:
    """Return a markdown risk badge based on a threshold comparison."""
    if reverse:
        ok = value >= high
        warn = low <= value < high
    else:
        ok = value <= low
        warn = low < value <= high
    if ok:
        return "🟢 Low"
    elif warn:
        return "🟡 Moderate"
    else:
        return "🔴 High"


def generate_report(eda_dir: Path):
    stats_json  = _load_json(eda_dir / "dataset_statistics.json")
    res_json    = _load_json(eda_dir / "resolution_summary.json")
    qual_json   = _load_json(eda_dir / "image_quality_analysis.json")

    lines = [
        "# EDA Observations and Conclusions",
        "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Dataset Overview",
        "",
    ]

    if stats_json:
        s = stats_json.get("summary", {})
        lines += [
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Images | **{s.get('total_images', 'N/A')}** |",
            f"| Total Classes | **{s.get('num_classes', 'N/A')}** |",
            f"| Mean Images / Class | {s.get('mean_per_class', 'N/A')} |",
            f"| Max / Class | {s.get('max_per_class', 'N/A')} |",
            f"| Min / Class | {s.get('min_per_class', 'N/A')} |",
            "",
        ]
    else:
        lines += ["> ⚠️ `dataset_statistics.json` not found — run `eda_statistics.py` first.", ""]

    # --- Class imbalance ---
    lines += ["## 2. Class Imbalance", ""]
    if stats_json:
        s = stats_json.get("summary", {})
        ir  = s.get("imbalance_ratio", 0)
        sev = s.get("imbalance_severity", "Unknown")
        gc  = s.get("gini_coefficient", 0)
        lines += [
            f"| Metric | Value | Risk |",
            f"|--------|-------|------|",
            f"| Imbalance Ratio (max/min) | **{ir}x** | {_risk_badge(ir, 2, 10)} |",
            f"| Gini Coefficient | {gc} | {_risk_badge(gc, 0.2, 0.5)} |",
            f"| Severity | **{sev}** | — |",
            "",
        ]

        maj = s.get("majority_classes", [])
        mino = s.get("minority_classes", [])
        if maj:
            lines.append("**Majority classes (risk: overfitting):**")
            for m in maj:
                lines.append(f"- {m['class']}: {m['count']} images")
        if mino:
            lines += ["", "**Minority classes (risk: underfitting / misclassification):**"]
            for m in mino:
                lines.append(f"- {m['class']}: {m['count']} images")

        if ir > 10:
            lines += [
                "",
                "> ⚠️ **Finding:** Severe class imbalance detected.",
                "> The model is likely to be biased toward majority classes.",
                "> **Recommended actions:**",
                "> - Apply class-weighted cross-entropy loss",
                "> - Augment minority-class images (flips, rotations, brightness jitter)",
                "> - Consider oversampling (SMOTE) or undersampling majority classes",
            ]
        elif ir > 3:
            lines += [
                "",
                "> ⚡ **Finding:** Moderate imbalance. Class weighting during training is advisable.",
            ]
        else:
            lines += ["", "> ✅ **Finding:** Dataset is reasonably balanced."]
    lines.append("")

    # --- Dimension analysis ---
    lines += ["## 3. Image Dimensions and Resolution", ""]
    if res_json:
        w = res_json.get("width", {})
        h = res_json.get("height", {})
        ar = res_json.get("aspect_ratio", {})
        same = res_json.get("all_same_size", False)
        uniq = res_json.get("unique_dimensions", 0)
        lines += [
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Width range  | {w.get('min')} – {w.get('max')} px (mean {w.get('mean')}) |",
            f"| Height range | {h.get('min')} – {h.get('max')} px (mean {h.get('mean')}) |",
            f"| Aspect ratio | {ar.get('min')} – {ar.get('max')} (mean {ar.get('mean')}) |",
            f"| Unique dimensions | {uniq} |",
            f"| All same size | {'✅ Yes' if same else '❌ No — resizing required'} |",
            "",
        ]
        if not same:
            lines += [
                "> ⚠️ **Finding:** Images have inconsistent dimensions.",
                "> All images must be resized to a fixed dimension before training.",
                "> **Recommended:** 224×224 for ImageNet-pretrained models.",
                "",
            ]
    else:
        lines += ["> ⚠️ `resolution_summary.json` not found — run `eda_image_dimensions.py` first.", ""]

    # --- Image quality ---
    lines += ["## 4. Image Quality Analysis", ""]
    if qual_json:
        flags    = qual_json.get("quality_flags", {})
        flag_pct = qual_json.get("flag_percentages", {})
        total    = qual_json.get("readable", 1)
        bri      = qual_json.get("brightness", {})
        blur     = qual_json.get("blur", {})
        sat      = qual_json.get("saturation", {})

        lines += [
            f"| Quality Issue | Count | % of Dataset | Risk |",
            f"|---------------|-------|--------------|------|",
        ]
        for issue, count in flags.items():
            pct = flag_pct.get(issue, 0)
            lines.append(
                f"| {issue} | {count} | {pct}% | {_risk_badge(pct, 5, 20)} |"
            )

        lines += [""]
        if bri:
            lines.append(
                f"**Brightness:** mean = {bri.get('mean')} ± {bri.get('std')}  "
                f"(range {bri.get('min')}–{bri.get('max')})"
            )
        if blur:
            lines.append(f"**Blur score:** mean = {blur.get('mean')} ± {blur.get('std')}")
        if sat:
            lines.append(f"**Saturation:** mean = {sat.get('mean')} ± {sat.get('std')}")

        # Specific observations
        lines += ["", "### Specific Observations", ""]
        dark_pct  = flag_pct.get("Dark / Under-exposed", 0)
        blur_pct  = flag_pct.get("Blurry", 0)
        bri_pct   = flag_pct.get("Over-exposed", 0)
        lc_pct    = flag_pct.get("Low Contrast", 0)

        if dark_pct > 10:
            lines.append(
                f"- 🔴 **Illumination problems:** {dark_pct:.1f}% of images are under-exposed. "
                "Underwater lighting conditions are inconsistent. "
                "Consider brightness augmentation during training."
            )
        if blur_pct > 10:
            lines.append(
                f"- 🔴 **Motion blur / turbidity:** {blur_pct:.1f}% of images scored below blur threshold. "
                "Fish movement and water turbidity cause blurry frames. "
                "Consider filtering extreme blur cases or applying sharpening augmentation."
            )
        if bri_pct > 5:
            lines.append(
                f"- 🟡 **Over-exposure:** {bri_pct:.1f}% of images are over-bright. "
                "Could be caused by camera flash near the lens or water surface reflections."
            )
        if lc_pct > 15:
            lines.append(
                f"- 🟡 **Low contrast:** {lc_pct:.1f}% of images have low contrast. "
                "May reduce feature discriminability. Consider contrast normalisation."
            )
        if dark_pct <= 5 and blur_pct <= 5 and bri_pct <= 5 and lc_pct <= 10:
            lines.append("- ✅ Image quality is generally good with no dominant issues.")
    else:
        lines += ["> ⚠️ `image_quality_analysis.json` not found — run `eda_image_quality.py` first.", ""]

    # --- Challenge summary ---
    lines += [
        "",
        "## 5. Potential Challenges Affecting Model Performance",
        "",
        "| # | Challenge | Severity | Mitigation |",
        "|---|-----------|----------|------------|",
        "| 1 | Class imbalance | 🔴 High | Weighted loss, oversampling, augmentation |",
        "| 2 | Variable image dimensions | 🟡 Medium | Resize to 224×224 (Phase 3) |",
        "| 3 | Motion blur / turbidity | 🟡 Medium | Filter or augment with blur |",
        "| 4 | Illumination variability | 🟡 Medium | Brightness / contrast augmentation |",
        "| 5 | Colour distortion (water column) | 🟡 Medium | ImageNet normalisation, colour jitter |",
        "| 6 | Occlusion (partial fish visibility) | 🟡 Medium | Random erasing augmentation |",
        "| 7 | Near-duplicate frames from video | 🔴 High | Leakage-safe splitting (Phase 2) ✅ Done |",
        "",
        "## 6. EDA Conclusions and Recommendations",
        "",
        "1. **Dataset acquisition:** Successfully collected underwater fish species dataset.",
        "2. **Class imbalance** is the most critical challenge — must be addressed during training.",
        "3. **Image quality** varies due to underwater conditions; augmentation is essential.",
        "4. **Leakage** has already been addressed in Phase 2 (video-group splitting).",
        "5. **Preprocessing** (Phase 3) handles dimension normalisation and pixel-level issues.",
        "6. **Recommended training strategy:**",
        "   - Transfer learning from ImageNet pretrained backbone",
        "   - Class-weighted cross-entropy loss",
        "   - Augmentation: horizontal/vertical flips, brightness jitter, Gaussian blur, random erasing",
        "   - Early stopping on validation loss",
        "",
        "---",
        "*Auto-generated by `eda_observations.py`*",
    ]

    out_path = eda_dir / "eda_observations.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] EDA observations → {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Compile EDA observations and conclusions")
    parser.add_argument("--eda-dir", default="reports/eda", help="EDA reports directory")
    args = parser.parse_args()

    eda_dir = Path(args.eda_dir)
    eda_dir.mkdir(parents=True, exist_ok=True)
    generate_report(eda_dir)


if __name__ == "__main__":
    main()
