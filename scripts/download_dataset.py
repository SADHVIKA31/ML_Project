"""
download_dataset.py
-------------------
Phase 1 – Step 1 & 2: Download and extract the Fish4Knowledge dataset
(or a comparable publicly available underwater-species dataset).

The Fish4Knowledge dataset is available via Kaggle:
  https://www.kaggle.com/datasets/markdaniellampa/fish-dataset

Usage:
    python scripts/download_dataset.py --source kaggle --dest data/raw
    python scripts/download_dataset.py --source url   --url <direct-zip-url> --dest data/raw

Requirements:
    pip install kaggle tqdm requests
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_kaggle(dataset_slug: str, dest: Path) -> Path:
    """Download a Kaggle dataset using the official Kaggle CLI."""
    try:
        import kaggle  # noqa: F401 – just verify the package is installed
    except ImportError:
        sys.exit(
            "[ERROR] 'kaggle' package not found.\n"
            "Install it with: pip install kaggle\n"
            "Then set up ~/.kaggle/kaggle.json with your API credentials."
        )

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading '{dataset_slug}' from Kaggle -> {dest}")
    os.system(f"kaggle datasets download -d {dataset_slug} -p {dest} --unzip")
    print("[INFO] Kaggle download complete.")
    return dest


def _download_url(url: str, dest: Path) -> Path:
    """Download a dataset zip from a direct URL and extract it."""
    import requests
    from tqdm import tqdm

    dest.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1] or "dataset.zip"
    zip_path = dest / filename

    print(f"[INFO] Downloading {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=filename
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"[INFO] Extracting {zip_path} -> {dest}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    zip_path.unlink()
    print("[INFO] Extraction complete.")
    return dest


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download underwater species dataset")
    parser.add_argument(
        "--source",
        choices=["kaggle", "url"],
        default="kaggle",
        help="Download source (default: kaggle)",
    )
    parser.add_argument(
        "--dataset",
        default="markdaniellampa/fish-dataset",
        help="Kaggle dataset slug (user/dataset-name)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Direct download URL (used when --source url)",
    )
    parser.add_argument(
        "--dest",
        default="data/raw",
        help="Destination directory (default: data/raw)",
    )
    args = parser.parse_args()

    dest = Path(args.dest)

    if args.source == "kaggle":
        _download_kaggle(args.dataset, dest)
    elif args.source == "url":
        if not args.url:
            sys.exit("[ERROR] --url is required when --source url")
        _download_url(args.url, dest)

    print(f"\n[DONE] Dataset saved to: {dest.resolve()}")


if __name__ == "__main__":
    main()
