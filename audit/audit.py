import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

# ============================================================
# SETTINGS
# ============================================================

DATASET_ROOT = Path("../datasets/extracted")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTENSIONS = {".txt"}
XML_EXTENSIONS = {".xml"}

# ============================================================
# FUNCTIONS
# ============================================================

def find_images(folder):
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def find_files(folder, extensions):
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]


def audit_yolo_labels(label_files):
    class_counter = Counter()
    total_boxes = 0
    empty_labels = 0
    invalid_labels = 0

    for file in label_files:
        try:
            lines = file.read_text(errors="ignore").splitlines()

            if not lines:
                empty_labels += 1
                continue

            valid_boxes = 0

            for line in lines:
                parts = line.strip().split()

                if len(parts) < 5:
                    invalid_labels += 1
                    continue

                try:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]

                    # Basic YOLO coordinate check
                    if not all(0 <= x <= 1 for x in coords):
                        invalid_labels += 1
                        continue

                    class_counter[class_id] += 1
                    total_boxes += 1
                    valid_boxes += 1

                except ValueError:
                    invalid_labels += 1

            if valid_boxes == 0:
                empty_labels += 1

        except Exception:
            invalid_labels += 1

    return class_counter, total_boxes, empty_labels, invalid_labels


def audit_xml_labels(xml_files):
    class_counter = Counter()
    total_boxes = 0
    invalid_xml = 0

    for file in xml_files:
        try:
            root = ET.parse(file).getroot()

            for obj in root.findall(".//object"):
                name = obj.find("name")

                if name is not None:
                    class_name = name.text.strip()
                    class_counter[class_name] += 1
                    total_boxes += 1

        except Exception:
            invalid_xml += 1

    return class_counter, total_boxes, invalid_xml


def find_json_files(folder):
    return [
        p for p in folder.rglob("*.json")
        if p.is_file()
    ]


def audit_coco_json(json_files):
    results = []

    for file in json_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "images" in data and "annotations" in data:
                categories = {
                    c["id"]: c["name"]
                    for c in data.get("categories", [])
                }

                class_counter = Counter()

                for ann in data.get("annotations", []):
                    category_id = ann.get("category_id")

                    if category_id in categories:
                        class_counter[categories[category_id]] += 1

                results.append({
                    "file": file,
                    "images": len(data.get("images", [])),
                    "annotations": len(data.get("annotations", [])),
                    "classes": class_counter
                })

        except Exception:
            pass

    return results


# ============================================================
# MAIN AUDIT
# ============================================================

print("=" * 70)
print("              DRONE / BIRD DATASET AUDIT")
print("=" * 70)

if not DATASET_ROOT.exists():
    print("\nERROR: Dataset folder not found:")
    print(DATASET_ROOT.resolve())
    raise SystemExit

datasets = [
    p for p in DATASET_ROOT.iterdir()
    if p.is_dir()
]

if not datasets:
    print("\nNo extracted datasets found.")
    raise SystemExit


for dataset in sorted(datasets):

    print("\n" + "=" * 70)
    print(f"DATASET: {dataset.name}")
    print("=" * 70)

    # -------------------------
    # Images
    # -------------------------

    images = find_images(dataset)

    print(f"\nImages: {len(images)}")

    # -------------------------
    # YOLO TXT
    # -------------------------

    txt_labels = find_files(dataset, LABEL_EXTENSIONS)

    print(f"TXT label files: {len(txt_labels)}")

    if txt_labels:
        classes, boxes, empty, invalid = audit_yolo_labels(txt_labels)

        print("\nYOLO annotations:")
        print(f"Total bounding boxes: {boxes}")
        print(f"Empty label files: {empty}")
        print(f"Invalid labels: {invalid}")

        print("\nClass IDs:")
        if classes:
            for class_id, count in sorted(classes.items()):
                print(f"  Class {class_id}: {count}")
        else:
            print("  No valid classes found.")

    # -------------------------
    # XML
    # -------------------------

    xml_labels = find_files(dataset, XML_EXTENSIONS)

    print(f"\nXML label files: {len(xml_labels)}")

    if xml_labels:
        classes, boxes, invalid = audit_xml_labels(xml_labels)

        print("\nPascal VOC annotations:")
        print(f"Total bounding boxes: {boxes}")
        print(f"Invalid XML files: {invalid}")

        print("\nClasses:")

        for class_name, count in sorted(classes.items()):
            print(f"  {class_name}: {count}")

    # -------------------------
    # COCO JSON
    # -------------------------

    json_files = find_json_files(dataset)

    coco_results = audit_coco_json(json_files)

    if coco_results:

        print("\nCOCO annotation files:")

        for result in coco_results:

            print(f"\n  File: {result['file'].name}")
            print(f"  Images: {result['images']}")
            print(f"  Annotations: {result['annotations']}")

            print("  Classes:")

            for class_name, count in result["classes"].items():
                print(f"    {class_name}: {count}")

    # -------------------------
    # Dataset structure
    # -------------------------

    print("\nTop-level folders:")

    for item in sorted(dataset.iterdir()):
        if item.is_dir():
            print(f"  [DIR]  {item.name}")
        else:
            print(f"  [FILE] {item.name}")


print("\n" + "=" * 70)
print("                    AUDIT COMPLETE")
print("=" * 70)