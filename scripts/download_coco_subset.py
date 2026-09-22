"""
COCO 2017 Mini-Subset Downloader for Vision Detection Evaluation.

Downloads a lightweight, representative subset of COCO 2017 validation images
restricted strictly to classes relevant to the Adaptive Multimodal HRI Framework:
- bottle
- cup
- laptop
- person
- chair
- couch (sofa)
- potted plant

Preserves ground-truth bounding box annotations for calculating Precision, Recall,
IoU, mAP@50, and mAP@50:95 without downloading multi-GB archives unnecessarily.
"""

import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Set

TARGET_CLASSES = {
    "person",
    "bottle",
    "cup",
    "chair",
    "couch",
    "potted plant",
    "laptop",
}

COCO_VAL_IMAGE_URL_PREFIX = "http://images.cocodataset.org/val2017/"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

OUTPUT_DIR = Path("data/coco_subset")
IMAGES_DIR = OUTPUT_DIR / "images"
ANNOTATIONS_FILE = OUTPUT_DIR / "annotations_subset.json"
NUM_IMAGES_PER_CLASS_TARGET = 8  # Balanced sampling targeting ~50-60 unique images


def download_and_extract_annotations(tmp_dir: Path) -> Path:
    """Download COCO annotations zip and extract instances_val2017.json."""
    zip_path = tmp_dir / "annotations_trainval2017.zip"
    val_json_path = tmp_dir / "instances_val2017.json"

    print(f"Downloading COCO annotations archive (~241MB)...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(COCO_ANNOTATIONS_URL, headers=headers)
    with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out_file:
        shutil.copyfileobj(resp, out_file)

    print("Extracting instances_val2017.json...")
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("annotations/instances_val2017.json") as src, open(val_json_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    # Immediately delete the large zip to conserve disk space
    if zip_path.exists():
        zip_path.unlink()
        print("Deleted temporary zip archive to conserve disk space.")

    return val_json_path


def create_curated_subset(val_json_path: Path, max_images: int = 50) -> Dict:
    """Filter COCO annotations to the target classes and select a curated image subset."""
    print("Loading instances_val2017.json...")
    with open(val_json_path, "r") as f:
        coco_data = json.load(f)

    # Map categories
    cat_id_to_name = {c["id"]: c["name"] for c in coco_data["categories"]}
    target_cat_ids = {c["id"]: c["name"] for c in coco_data["categories"] if c["name"] in TARGET_CLASSES}
    print(f"Target categories found: {list(target_cat_ids.values())}")

    # Group annotations by image_id
    img_to_anns = {}
    for ann in coco_data["annotations"]:
        cat_id = ann["category_id"]
        if cat_id in target_cat_ids:
            img_id = ann["image_id"]
            img_to_anns.setdefault(img_id, []).append(ann)

    # Balanced selection of images across target classes
    selected_img_ids: Set[int] = set()
    class_counts = {name: 0 for name in TARGET_CLASSES}

    # First pass: try to balance classes
    for cat_id, cat_name in target_cat_ids.items():
        count_for_cat = 0
        for img_id, anns in img_to_anns.items():
            if any(a["category_id"] == cat_id for a in anns):
                selected_img_ids.add(img_id)
                count_for_cat += 1
                if count_for_cat >= NUM_IMAGES_PER_CLASS_TARGET:
                    break

    # Build image map
    all_images_map = {img["id"]: img for img in coco_data["images"]}
    selected_images = [all_images_map[img_id] for img_id in selected_img_ids if img_id in all_images_map]
    
    # Cap total images if exceeds max_images
    if len(selected_images) > max_images:
        selected_images = selected_images[:max_images]
        selected_img_ids = {img["id"] for img in selected_images}

    # Filter annotations for selected images
    selected_annotations = []
    for ann in coco_data["annotations"]:
        if ann["image_id"] in selected_img_ids and ann["category_id"] in target_cat_ids:
            selected_annotations.append(ann)
            cat_name = cat_id_to_name[ann["category_id"]]
            class_counts[cat_name] = class_counts.get(cat_name, 0) + 1

    subset_coco = {
        "info": {
            "description": "Curated COCO 2017 Validation Mini-Subset for HRI Vision Evaluation",
            "version": "1.0",
            "year": 2026,
            "target_classes": list(TARGET_CLASSES),
        },
        "licenses": coco_data.get("licenses", []),
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": [c for c in coco_data["categories"] if c["id"] in target_cat_ids],
    }

    print(f"\nSubset Summary:")
    print(f"  Total unique images: {len(selected_images)}")
    print(f"  Total ground-truth bounding boxes: {len(selected_annotations)}")
    print("  Ground-truth box counts per class:")
    for cname, cnt in sorted(class_counts.items()):
        print(f"    - {cname:<15}: {cnt}")

    return subset_coco


def download_subset_images(images: List[Dict], dest_dir: Path):
    """Download individual images from COCO repository."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    total = len(images)
    print(f"\nDownloading {total} individual image files into {dest_dir}...")

    headers = {"User-Agent": "Mozilla/5.0"}
    for idx, img in enumerate(images, 1):
        file_name = img["file_name"]
        file_path = dest_dir / file_name
        if file_path.exists():
            continue

        url = f"{COCO_VAL_IMAGE_URL_PREFIX}{file_name}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp, open(file_path, "wb") as f:
            shutil.copyfileobj(resp, f)

        if idx % 10 == 0 or idx == total:
            print(f"  [{idx}/{total}] Downloaded {file_name}")

    print("All subset images downloaded successfully.")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("data/tmp_coco")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        val_json_path = download_and_extract_annotations(tmp_dir)
        subset_coco = create_curated_subset(val_json_path, max_images=50)

        # Save subset annotations JSON
        with open(ANNOTATIONS_FILE, "w") as f:
            json.dump(subset_coco, f, indent=2)
        print(f"\nSaved curated ground-truth annotations to {ANNOTATIONS_FILE}")

        # Download only the subset images
        download_subset_images(subset_coco["images"], IMAGES_DIR)

    finally:
        # Clean up temporary extraction folder
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
            print("Cleaned up temporary download directory.")

    # Calculate total size on disk
    total_size = sum(f.stat().st_size for f in IMAGES_DIR.glob("*.*")) + ANNOTATIONS_FILE.stat().st_size
    print(f"\nTotal disk space used by COCO subset: {total_size / (1024 * 1024):.2f} MB")
    print("Done!")


if __name__ == "__main__":
    main()
