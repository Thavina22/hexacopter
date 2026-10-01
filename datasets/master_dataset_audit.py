from pathlib import Path
from collections import Counter
import json
import xml.etree.ElementTree as ET
import yaml

ROOT = Path(r".\datasets\raw")

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}


# ============================================================
# HELPERS
# ============================================================

def images_in(folder):
    if not folder.exists():
        return []

    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def txt_files(folder):
    if not folder.exists():
        return []

    return list(folder.rglob("*.txt"))


def xml_files(folder):
    if not folder.exists():
        return []

    return list(folder.rglob("*.xml"))


def json_files(folder):
    if not folder.exists():
        return []

    return list(folder.rglob("*.json"))


def print_header(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# ============================================================
# BIRD
# ============================================================

def audit_bird():

    root = (
        ROOT
        / "bird-detection-yolo-dataset-main"
        / "bird-detection-yolo-dataset-main"
    )

    print_header("1. BIRD DATASET — NEW")

    if not root.exists():
        print("NOT FOUND")
        return

    yaml_files = list(root.glob("*.yaml")) + list(root.glob("*.yml"))

    print(f"Root: {root}")
    print()

    if yaml_files:
        for y in yaml_files:
            print(f"YAML: {y}")
            try:
                data = yaml.safe_load(y.read_text(encoding="utf-8"))
                print(f"Classes: {data.get('names')}")
            except Exception as e:
                print(f"YAML read error: {e}")

    total_images = 0
    total_boxes = 0
    valid_images = 0
    empty = 0
    invalid = 0
    classes = Counter()

    for split in ["train", "val"]:

        image_dir = root / "images" / split
        label_dir = root / "labels" / split

        images = images_in(image_dir)

        print()
        print(f"{split.upper()}")
        print("-" * 60)
        print(f"Images: {len(images)}")

        split_boxes = 0
        split_empty = 0

        for image in images:

            label = label_dir / f"{image.stem}.txt"

            if not label.exists():
                continue

            lines = [
                x.strip()
                for x in label.read_text(
                    encoding="utf-8",
                    errors="ignore"
                ).splitlines()
                if x.strip()
            ]

            if not lines:
                split_empty += 1
                empty += 1
                continue

            valid_this_image = True

            for line in lines:

                parts = line.split()

                if len(parts) != 5:
                    valid_this_image = False
                    invalid += 1
                    continue

                try:
                    cls = int(float(parts[0]))
                    x, y, w, h = map(float, parts[1:])

                except ValueError:
                    valid_this_image = False
                    invalid += 1
                    continue

                if not (
                    0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < w <= 1
                    and 0 < h <= 1
                ):
                    valid_this_image = False
                    invalid += 1
                    continue

                classes[cls] += 1
                split_boxes += 1

            if valid_this_image:
                valid_images += 1

        print(f"Boxes: {split_boxes}")
        print(f"Empty labels: {split_empty}")

        total_images += len(images)
        total_boxes += split_boxes

    print()
    print("BIRD SUMMARY")
    print("-" * 60)
    print(f"Images             : {total_images}")
    print(f"Usable images      : {valid_images}")
    print(f"Boxes              : {total_boxes}")
    print(f"Empty labels       : {empty}")
    print(f"Invalid annotations: {invalid}")
    print(f"Original classes   : {dict(classes)}")


# ============================================================
# NEW UAV VOC
# ============================================================

def audit_new_uav():

    root = (
        ROOT
        / "uav-20261001T061724Z-1-001"
        / "uav"
    )

    print_header("2. NEW UAV DATASET — VOC")

    if not root.exists():
        print("NOT FOUND")
        return

    image_dir = root / "JPEGImages"
    xml_dir = root / "Annotations"

    images = images_in(image_dir)
    xmls = xml_files(xml_dir)

    image_map = {
        p.stem.lower(): p
        for p in images
    }

    xml_map = {
        p.stem.lower(): p
        for p in xmls
    }

    paired = set(image_map) & set(xml_map)

    classes = Counter()
    boxes = 0
    invalid = 0

    for xml in xmls:

        try:
            tree = ET.parse(xml)
            root_xml = tree.getroot()
        except Exception:
            invalid += 1
            continue

        for obj in root_xml.findall("object"):

            name = obj.findtext("name")

            if not name:
                continue

            classes[name.strip()] += 1

            bbox = obj.find("bndbox")

            if bbox is None:
                invalid += 1
                continue

            try:
                xmin = float(bbox.findtext("xmin"))
                ymin = float(bbox.findtext("ymin"))
                xmax = float(bbox.findtext("xmax"))
                ymax = float(bbox.findtext("ymax"))

                if xmax <= xmin or ymax <= ymin:
                    invalid += 1
                    continue

                boxes += 1

            except Exception:
                invalid += 1

    print(f"Images              : {len(images)}")
    print(f"XML annotations     : {len(xmls)}")
    print(f"Image/XML pairs     : {len(paired)}")
    print(f"Images without XML  : {len(set(image_map) - set(xml_map))}")
    print(f"XML without image   : {len(set(xml_map) - set(image_map))}")
    print(f"Boxes               : {boxes}")
    print(f"Invalid             : {invalid}")
    print(f"Classes             : {dict(classes)}")


# ============================================================
# CLOUD DRONE
# ============================================================

def audit_cloud():

    root = ROOT / "cloud_drone_aug"

    print_header("3. CLOUD DRONE AUG — PREVIOUS")

    if not root.exists():
        print("NOT FOUND")
        return

    images = images_in(root)

    # Only look for labels inside likely label directories
    labels = [
        p for p in txt_files(root)
        if p.name.lower() not in {
            "classes.txt",
            "requirements.txt"
        }
    ]

    boxes = 0
    empty = 0
    invalid = 0
    classes = Counter()

    for label in labels:

        lines = [
            x.strip()
            for x in label.read_text(
                encoding="utf-8",
                errors="ignore"
            ).splitlines()
            if x.strip()
        ]

        if not lines:
            empty += 1
            continue

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                invalid += 1
                continue

            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])

            except ValueError:
                invalid += 1
                continue

            if not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                invalid += 1
                continue

            classes[cls] += 1
            boxes += 1

    print(f"Images         : {len(images)}")
    print(f"TXT labels     : {len(labels)}")
    print(f"Boxes          : {boxes}")
    print(f"Empty labels   : {empty}")
    print(f"Invalid        : {invalid}")
    print(f"Classes        : {dict(classes)}")


# ============================================================
# PREVIOUS UAVs COCO
# ============================================================

def audit_uavs():

    root = ROOT / "UAVs"

    print_header("4. UAVs — PREVIOUS COCO DATASET")

    if not root.exists():
        print("NOT FOUND")
        return

    jsons = json_files(root)

    print(f"JSON files found: {len(jsons)}")

    total_images = 0
    total_annotations = 0
    categories = Counter()

    for js in jsons:

        name = js.name.lower()

        if not (
            "train" in name
            or "valid" in name
            or "test" in name
        ):
            continue

        try:
            data = json.loads(
                js.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        cats = data.get("categories", [])

        total_images += len(images)
        total_annotations += len(annotations)

        category_map = {
            c["id"]: c["name"]
            for c in cats
        }

        for ann in annotations:
            cid = ann.get("category_id")
            categories[category_map.get(cid, str(cid))] += 1

        print()
        print(js.name)
        print(f"  Images      : {len(images)}")
        print(f"  Annotations : {len(annotations)}")
        print(f"  Categories  : {category_map}")

    print()
    print("UAVs TOTAL")
    print("-" * 60)
    print(f"Images       : {total_images}")
    print(f"Annotations  : {total_annotations}")
    print(f"Classes      : {dict(categories)}")


# ============================================================
# MASTER
# ============================================================

def main():

    print_header("MASTER DRONE + BIRD DATASET AUDIT")

    print(f"ROOT: {ROOT.resolve()}")

    audit_bird()
    audit_new_uav()
    audit_cloud()
    audit_uavs()

    print_header("AUDIT COMPLETE")

    print("""
CANDIDATE TRAINING SOURCES

BIRD:
  bird-detection-yolo-dataset-main

DRONE:
  uav-20261001T061724Z-1-001
  cloud_drone_aug
  UAVs

NOT CURRENTLY INCLUDED:
  TIB-Net-master
  VisDrone-Dataset-master

IMPORTANT:
  This script only audits.
  It does NOT copy, delete, modify, or merge datasets.
""")


if __name__ == "__main__":
    main()