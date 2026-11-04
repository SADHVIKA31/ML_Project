# Phase 4 – Exploratory Data Analysis (EDA): Methodology

> **Branch:** `sadhvika_phase4`
> **Responsibility:** Sadhvika – Dataset & Preprocessing Lead

---

## Pipeline Overview

```
data/processed/            (Phase 1-3 output)
        │
        ├─► eda_statistics.py          Tasks 1,2,5  → dataset_statistics.{json,md}
        │
        ├─► eda_class_distribution.py  Tasks 3,4    → class_distribution_bar.png
        │                                            → class_distribution_pie.png
        │                                            → class_imbalance_heatmap.png
        │
        ├─► eda_image_dimensions.py    Task 6       → dimension_scatter.png
        │                                            → dimension_histogram.png
        │                                            → aspect_ratio_dist.png
        │                                            → resolution_summary.json
        │
        ├─► eda_sample_images.py       Task 7       → sample_images_grid.png
        │                                            → sample_per_class/*.png
        │
        ├─► eda_image_quality.py       Tasks 8-10   → image_quality_analysis.{csv,json}
        │                                            → quality_brightness_hist.png
        │                                            → quality_blur_hist.png
        │                                            → quality_flags_summary.png
        │
        └─► eda_observations.py        Task 11      → eda_observations.md
```

---

## Script-by-Script Methodology

### `eda_statistics.py` — Tasks 1, 2, 5

| What | How |
|------|-----|
| Total images | Recursive glob across all class subdirectories |
| Per-species count | Python `Counter` on class folder names |
| Imbalance ratio | `max_count / min_count` |
| Gini coefficient | Standard inequality measure (0 = balanced, 1 = all in one class) |
| Coefficient of variation | `std / mean` — relative spread of class sizes |
| Severity label | Classified as BALANCED / MILD / MODERATE / HIGH / SEVERE |

---

### `eda_class_distribution.py` — Tasks 3, 4

Three complementary visualisations:

| Chart | Description |
|-------|-------------|
| Horizontal bar chart | All classes ranked by image count; majority=green, minority=red |
| Pie chart | Top-10 classes + Others aggregate |
| Imbalance heatmap | Per-class deviation from mean, green = over-represented, red = under |

---

### `eda_image_dimensions.py` — Task 6

| Plot | Insight |
|------|---------|
| Width × Height scatter | Shows whether images cluster at a standard size |
| Width & height histograms | Distribution shape, outliers |
| Aspect ratio distribution | Identifies landscape/portrait/square dominance |

---

### `eda_sample_images.py` — Task 7

- **Main grid:** N samples per class side by side — visual taxonomy overview
- **Per-class grids:** More samples per class saved individually in `sample_per_class/`
- Random seed = 42 for reproducibility

---

### `eda_image_quality.py` — Tasks 8, 9, 10

| Metric | Method | Detects |
|--------|--------|---------|
| Brightness | Mean pixel intensity (0–255) | Dark (illumination) / over-exposed |
| Contrast | Std of pixel intensity | Low contrast / flat images |
| Blur score | Laplacian variance | Motion blur / turbidity |
| Saturation | Mean HSV S-channel | Colour distortion / desaturation |

**Default thresholds:**

| Flag | Condition |
|------|-----------|
| `is_dark` | brightness < 50 |
| `is_bright` | brightness > 220 |
| `is_blurry` | Laplacian variance < 100 |
| `is_low_contrast` | contrast std < 20 |

---

### `eda_observations.py` — Task 11

Reads all previously generated JSON reports and synthesises:
- Dataset overview table
- Class imbalance risk assessment (🟢🟡🔴 badges)
- Dimension inconsistency finding
- Quality flag interpretation (illumination, blur, turbidity)
- **Challenge table** with severity and mitigation strategy
- Final conclusions and training recommendations

---

## Reproducing the EDA

```bash
# 0. Install dependencies
pip install -r requirements.txt -r requirements_phase2.txt \
            -r requirements_phase3.txt -r requirements_phase4.txt

# 1. Dataset statistics + imbalance metrics
python scripts/eda_statistics.py --dataset data/processed --out reports/eda

# 2. Class distribution charts
python scripts/eda_class_distribution.py --dataset data/processed --out reports/eda

# 3. Image dimension analysis
python scripts/eda_image_dimensions.py --dataset data/processed --out reports/eda

# 4. Sample image visualisations
python scripts/eda_sample_images.py --dataset data/processed --out reports/eda --samples 4

# 5. Image quality analysis (use --max-images 1000 for speed on large datasets)
python scripts/eda_image_quality.py --dataset data/processed --out reports/eda

# 6. Compile observations and conclusions
python scripts/eda_observations.py --eda-dir reports/eda
```

---

## Expected Deliverables

| Deliverable | File |
|-------------|------|
| Class distribution bar chart | `reports/eda/class_distribution_bar.png` |
| Class distribution pie chart | `reports/eda/class_distribution_pie.png` |
| Imbalance heatmap | `reports/eda/class_imbalance_heatmap.png` |
| Sample image visualisations | `reports/eda/sample_images_grid.png` |
| Per-class sample grids | `reports/eda/sample_per_class/<class>_samples.png` |
| Dimension scatter | `reports/eda/dimension_scatter.png` |
| Dimension histograms | `reports/eda/dimension_histogram.png` |
| Aspect ratio chart | `reports/eda/aspect_ratio_dist.png` |
| Quality flags chart | `reports/eda/quality_flags_summary.png` |
| Dataset statistics | `reports/eda/dataset_statistics.{json,md}` |
| Resolution summary | `reports/eda/resolution_summary.json` |
| Quality analysis | `reports/eda/image_quality_analysis.{csv,json}` |
| EDA conclusions | `reports/eda/eda_observations.md` |

---

*Branch: `sadhvika_phase4` → PR target: `develop`*
