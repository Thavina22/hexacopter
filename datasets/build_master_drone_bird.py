from pathlib import Path
import shutil
import random
import hashlib
import xml.etree.ElementTree as ET
import yaml


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "datasets" / "raw"
EXTRACTED = ROOT / "datasets" / "extracted"

OUTPUT = ROOT / "datasets" / "yolo" / "drone_bird_master"

SEED = 42
random.seed(SEED)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Final classes
DRONE = 0
BIRD = 1


# ============================================================
# HELPERS
# ============================================================

def read_yolo_label(label_path):
    """
    Read YOLO labels.

    Returns:
        list of tuples:
        (class_id, x_center, y_center, width, height)
    """

    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            try:
                cls = int(float(parts[0]))
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue

            # Validate normalized YOLO coordinates
            if not (
                0 <= xc <= 1
                and 0 <= yc <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                continue

            boxes.append((cls, xc, yc, w, h))

    return boxes


def convert_boxes_to_class(boxes, new_class):
    """
    Change all source classes to the final class.
    """

    return [
        (new_class, xc, yc, w, h)
        for _, xc, yc, w, h in boxes
    ]


def write_yolo_label(path, boxes):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for cls, xc, yc, w, h in boxes:
            f.write(
                f"{cls} "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{w:.6f} "
                f"{h:.6f}\n"
            )


def image_hash(path):
    """
    Exact file hash used for duplicate detection.
    """

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def make_record(image, boxes, source, original_split=None):
    return {
        "image": Path(image),
        "boxes": boxes,
        "source": source,
        "original_split": original_split,
    }


# ============================================================
# BIRD DATASET
# ============================================================

def collect_birds():
    root = (
        RAW
        / "bird-detection-yolo-dataset-main"
        / "bird-detection-yolo-dataset-main"
    )

    records = []

    # Only train and val.
    # samples/ is intentionally excluded.
    for split in ["train", "val"]:

        image_dir = root / "images" / split
        label_dir = root / "labels" / split

        if not image_dir.exists():
            continue

        for image_path in image_dir.iterdir():

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                continue

            boxes = read_yolo_label(label_path)

            # Empty-label bird images were manually inspected
            # and contain birds, but have no usable annotations.
            # Therefore exclude them rather than treating them
            # as negatives.
            if not boxes:
                continue

            boxes = convert_boxes_to_class(boxes, BIRD)

            records.append(
                make_record(
                    image_path,
                    boxes,
                    "bird",
                    split,
                )
            )

    return records


# ============================================================
# CLOUD DRONE AUG
# ============================================================

def collect_cloud_drone():
    """
    Cloud Drone Aug structure:

    cloud_drone_aug/
        images/
            train/
            val/
            test/
        labels/
            train/
            val/
            test/
    """

    root = EXTRACTED / "cloud_drone_aug"

    records = []

    for split in ["train", "val", "test"]:

        image_dir = root / "images" / split
        label_dir = root / "labels" / split

        if not image_dir.exists():
            print(f"[WARN] Cloud image directory missing: {image_dir}")
            continue

        if not label_dir.exists():
            print(f"[WARN] Cloud label directory missing: {label_dir}")
            continue

        split_count = 0
        split_boxes = 0

        for image_path in image_dir.iterdir():

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                print(
                    f"[WARN] Cloud label missing: "
                    f"{split}/{image_path.name}"
                )
                continue

            boxes = read_yolo_label(label_path)

            if not boxes:
                continue

            boxes = convert_boxes_to_class(boxes, DRONE)

            records.append(
                make_record(
                    image_path,
                    boxes,
                    "cloud_drone_aug",
                    split,
                )
            )

            split_count += 1
            split_boxes += len(boxes)

        print(
            f"CLOUD {split:5s} -> "
            f"images={split_count} boxes={split_boxes}"
        )

    return records


# ============================================================
# NEW UAV VOC DATASET
# ============================================================

def collect_new_uav():
    """
    Structure:

    uav/
        JPEGImages/
        Annotations/
    """

    root = (
        RAW
        / "uav-20261001T061724Z-1-001"
        / "uav"
    )

    image_dir = root / "JPEGImages"
    annotation_dir = root / "Annotations"

    records = []

    if not image_dir.exists():
        print("[WARN] New UAV image directory missing")
        return records

    if not annotation_dir.exists():
        print("[WARN] New UAV annotation directory missing")
        return records

    for image_path in image_dir.iterdir():

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        xml_path = annotation_dir / f"{image_path.stem}.xml"

        # Explicitly exclude images without XML.
        if not xml_path.exists():
            continue

        try:
            tree = ET.parse(xml_path)
            root_xml = tree.getroot()
        except Exception:
            continue

        boxes = []

        size = root_xml.find("size")

        if size is None:
            continue

        width_node = size.find("width")
        height_node = size.find("height")

        if width_node is None or height_node is None:
            continue

        try:
            img_w = float(width_node.text)
            img_h = float(height_node.text)
        except (TypeError, ValueError):
            continue

        if img_w <= 0 or img_h <= 0:
            continue

        for obj in root_xml.findall("object"):

            bbox = obj.find("bndbox")

            if bbox is None:
                continue

            try:
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
            except (AttributeError, TypeError, ValueError):
                continue

            if xmax <= xmin or ymax <= ymin:
                continue

            # Clamp
            xmin = max(0, min(xmin, img_w))
            xmax = max(0, min(xmax, img_w))
            ymin = max(0, min(ymin, img_h))
            ymax = max(0, min(ymax, img_h))

            if xmax <= xmin or ymax <= ymin:
                continue

            xc = ((xmin + xmax) / 2) / img_w
            yc = ((ymin + ymax) / 2) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            boxes.append(
                (
                    DRONE,
                    xc,
                    yc,
                    w,
                    h,
                )
            )

        if not boxes:
            continue

        records.append(
            make_record(
                image_path,
                boxes,
                "new_uav",
                None,
            )
        )

    return records


# ============================================================
# EXISTING UAVs DATASET
# ============================================================

def collect_uavs():
    """
    Uses the already-converted UAVs YOLO dataset.
    """

    root = ROOT / "datasets" / "yolo" / "UAVs"

    records = []

    if not root.exists():
        print("[WARN] Converted UAVs dataset not found")
        return records

    for split in ["train", "val", "test"]:

        image_dir = root / "images" / split
        label_dir = root / "labels" / split

        if not image_dir.exists():
            continue

        for image_path in image_dir.iterdir():

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                continue

            boxes = read_yolo_label(label_path)

            if not boxes:
                continue

            boxes = convert_boxes_to_class(boxes, DRONE)

            records.append(
                make_record(
                    image_path,
                    boxes,
                    "UAVs",
                    split,
                )
            )

    return records


# ============================================================
# SPLITTING
# ============================================================

def split_records(records, train_ratio=0.8, val_ratio=0.1):
    """
    Randomly split records into train/val/test.
    """

    records = list(records)
    random.shuffle(records)

    n = len(records)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return (
        records[:train_end],
        records[train_end:val_end],
        records[val_end:],
    )


# ============================================================
# BUILD FINAL DATASET
# ============================================================

def main():

    print("=" * 70)
    print("MASTER DRONE + BIRD DATASET BUILDER")
    print("=" * 70)

    print(f"Output: {OUTPUT}")
    print()

    # --------------------------------------------------------
    # Collect
    # --------------------------------------------------------

    birds = collect_birds()
    cloud = collect_cloud_drone()
    new_uav = collect_new_uav()
    uavs = collect_uavs()

    print()
    print("SOURCE SUMMARY")
    print("-" * 70)

    print(
        f"Bird dataset       : "
        f"{len(birds):6d} images, "
        f"{sum(len(x['boxes']) for x in birds):6d} boxes"
    )

    print(
        f"Cloud Drone Aug    : "
        f"{len(cloud):6d} images, "
        f"{sum(len(x['boxes']) for x in cloud):6d} boxes"
    )

    print(
        f"New UAV            : "
        f"{len(new_uav):6d} images, "
        f"{sum(len(x['boxes']) for x in new_uav):6d} boxes"
    )

    print(
        f"UAVs               : "
        f"{len(uavs):6d} images, "
        f"{sum(len(x['boxes']) for x in uavs):6d} boxes"
    )

    # --------------------------------------------------------
    # Preserve existing train/val/test for UAVs and Cloud
    # --------------------------------------------------------

    train_records = []
    val_records = []
    test_records = []

    # Bird dataset:
    # original train is split 80/10/10
    # original val remains validation
    bird_train = [x for x in birds if x["original_split"] == "train"]
    bird_val = [x for x in birds if x["original_split"] == "val"]

    b_train, b_val, b_test = split_records(bird_train)

    train_records.extend(b_train)
    val_records.extend(b_val)
    test_records.extend(b_test)

    val_records.extend(bird_val)

    # Cloud:
    # preserve original split
    for r in cloud:
        if r["original_split"] == "train":
            train_records.append(r)
        elif r["original_split"] == "val":
            val_records.append(r)
        elif r["original_split"] == "test":
            test_records.append(r)

    # Existing UAVs:
    # preserve original split
    for r in uavs:
        if r["original_split"] == "train":
            train_records.append(r)
        elif r["original_split"] == "val":
            val_records.append(r)
        elif r["original_split"] == "test":
            test_records.append(r)

    # New UAV:
    # randomly split 80/10/10
    n_train, n_val, n_test = split_records(new_uav)

    train_records.extend(n_train)
    val_records.extend(n_val)
    test_records.extend(n_test)

    print()
    print("PRE-DEDUPE SPLITS")
    print("-" * 70)
    print(f"Train : {len(train_records)}")
    print(f"Val   : {len(val_records)}")
    print(f"Test  : {len(test_records)}")

    # --------------------------------------------------------
    # Exact duplicate removal
    # --------------------------------------------------------

    seen = set()

    final_train = []
    final_val = []
    final_test = []

    removed_train = 0
    removed_val = 0
    removed_test = 0

    # Train gets priority
    for r in train_records:

        try:
            h = image_hash(r["image"])
        except Exception:
            continue

        if h in seen:
            removed_train += 1
            continue

        seen.add(h)
        final_train.append(r)

    # Then validation
    for r in val_records:

        try:
            h = image_hash(r["image"])
        except Exception:
            continue

        if h in seen:
            removed_val += 1
            continue

        seen.add(h)
        final_val.append(r)

    # Then test
    for r in test_records:

        try:
            h = image_hash(r["image"])
        except Exception:
            continue

        if h in seen:
            removed_test += 1
            continue

        seen.add(h)
        final_test.append(r)

    print()
    print("EXACT DUPLICATE CHECK")
    print("-" * 70)
    print(f"Unique images      : {len(seen)}")
    print(f"Removed from train : {removed_train}")
    print(f"Removed from val   : {removed_val}")
    print(f"Removed from test  : {removed_test}")

    # --------------------------------------------------------
    # Remove previous output
    # --------------------------------------------------------

    if OUTPUT.exists():
        print()
        print(f"Removing previous dataset: {OUTPUT}")
        shutil.rmtree(OUTPUT)

    # --------------------------------------------------------
    # Write dataset
    # --------------------------------------------------------

    split_map = {
        "train": final_train,
        "val": final_val,
        "test": final_test,
    }

    source_stats = {}

    for split, records in split_map.items():

        image_dir = OUTPUT / "images" / split
        label_dir = OUTPUT / "labels" / split

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for index, r in enumerate(records):

            source = r["source"]

            source_stats.setdefault(
                source,
                {
                    "images": 0,
                    "boxes": 0,
                    "drone": 0,
                    "bird": 0,
                },
            )

            # Unique filename
            new_name = (
                f"{source}_{index:07d}"
                + r["image"].suffix.lower()
            )

            destination_image = image_dir / new_name
            destination_label = label_dir / (
                f"{source}_{index:07d}.txt"
            )

            shutil.copy2(
                r["image"],
                destination_image,
            )

            write_yolo_label(
                destination_label,
                r["boxes"],
            )

            source_stats[source]["images"] += 1
            source_stats[source]["boxes"] += len(r["boxes"])

            for box in r["boxes"]:
                if box[0] == DRONE:
                    source_stats[source]["drone"] += 1
                elif box[0] == BIRD:
                    source_stats[source]["bird"] += 1

    # --------------------------------------------------------
    # Dataset YAML
    # --------------------------------------------------------

    data_yaml = {
        "path": str(OUTPUT.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "drone",
            1: "bird",
        },
    }

    yaml_path = OUTPUT / "data.yaml"

    with open(
        yaml_path,
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(
            data_yaml,
            f,
            sort_keys=False,
        )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    total_images = (
        len(final_train)
        + len(final_val)
        + len(final_test)
    )

    total_boxes = sum(
        stats["boxes"]
        for stats in source_stats.values()
    )

    total_drone = sum(
        stats["drone"]
        for stats in source_stats.values()
    )

    total_bird = sum(
        stats["bird"]
        for stats in source_stats.values()
    )

    print()
    print("=" * 70)
    print("FINAL MASTER DATASET")
    print("=" * 70)

    print(
        f"TRAIN : {len(final_train):6d} images"
    )

    print(
        f"VAL   : {len(final_val):6d} images"
    )

    print(
        f"TEST  : {len(final_test):6d} images"
    )

    print(
        f"TOTAL : {total_images:6d} images"
    )

    print()
    print(
        f"TOTAL BOXES : {total_boxes}"
    )

    print(
        f"DRONE BOXES : {total_drone}"
    )

    print(
        f"BIRD BOXES  : {total_bird}"
    )

    print()
    print("SOURCE CONTRIBUTION")
    print("-" * 70)

    for source, stats in source_stats.items():

        print(
            f"{source:20s} "
            f"images={stats['images']:6d} "
            f"boxes={stats['boxes']:6d} "
            f"drone={stats['drone']:6d} "
            f"bird={stats['bird']:6d}"
        )

    print()
    print("CLASS MAPPING")
    print("-" * 70)
    print("0 = drone")
    print("1 = bird")

    print()
    print(f"YAML: {yaml_path}")

    print()
    print("=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()