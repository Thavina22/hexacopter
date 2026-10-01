from pathlib import Path
from collections import Counter
import random
import shutil
import xml.etree.ElementTree as ET

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r".\datasets\raw")
OUT = Path(r".\datasets\yolo\drone_bird")

SEED = 42

# UAV split
UAV_TRAIN = 0.80
UAV_VAL = 0.10
UAV_TEST = 0.10

# Bird train split
# Existing bird validation remains validation.
# 90% of valid bird training images -> train
# 10% of valid bird training images -> test
BIRD_TRAIN = 0.90
BIRD_TEST = 0.10

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# PATHS
# ============================================================

BIRD_ROOT = (
    ROOT
    / "bird-detection-yolo-dataset-main"
    / "bird-detection-yolo-dataset-main"
)

UAV_ROOT = (
    ROOT
    / "uav-20261001T061724Z-1-001"
    / "uav"
)

BIRD_IMAGE_ROOT = BIRD_ROOT / "images"
BIRD_LABEL_ROOT = BIRD_ROOT / "labels"

UAV_IMAGE_ROOT = UAV_ROOT / "JPEGImages"
UAV_XML_ROOT = UAV_ROOT / "Annotations"


# ============================================================
# HELPERS
# ============================================================

def reset_output():

    if OUT.exists():
        print(f"Removing old output: {OUT}")
        shutil.rmtree(OUT)

    for split in ["train", "val", "test"]:

        (OUT / "images" / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (OUT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True
        )


def copy_pair(image, label_text, split, prefix):

    new_name = f"{prefix}_{image.stem}{image.suffix.lower()}"

    dst_image = (
        OUT
        / "images"
        / split
        / new_name
    )

    dst_label = (
        OUT
        / "labels"
        / split
        / f"{prefix}_{image.stem}.txt"
    )

    shutil.copy2(image, dst_image)

    dst_label.write_text(
        label_text,
        encoding="utf-8"
    )


def read_yolo_label(path):

    lines = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()
        if line.strip()
    ]

    return lines


def validate_yolo_lines(lines):

    valid = []

    for line in lines:

        parts = line.split()

        if len(parts) != 5:
            return None

        try:
            class_id = int(float(parts[0]))

            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except ValueError:
            return None

        if not (
            0 <= x <= 1 and
            0 <= y <= 1 and
            0 < w <= 1 and
            0 < h <= 1
        ):
            return None

        # Original bird dataset says class 0 = bird.
        # Convert to final class 1.
        if class_id != 0:
            return None

        valid.append(
            f"1 {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        )

    return valid


def find_image(folder, stem):

    for ext in IMAGE_EXTS:

        candidate = folder / f"{stem}{ext}"

        if candidate.exists():
            return candidate

    return None


# ============================================================
# COLLECT BIRD DATA
# ============================================================

def collect_bird_split(split):

    image_dir = BIRD_IMAGE_ROOT / split
    label_dir = BIRD_LABEL_ROOT / split

    records = []

    if not image_dir.exists():
        print(f"Missing bird image directory: {image_dir}")
        return records

    if not label_dir.exists():
        print(f"Missing bird label directory: {label_dir}")
        return records

    images = sorted(
        [
            p for p in image_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
        ],
        key=lambda p: p.name.lower()
    )

    for image in images:

        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            continue

        lines = read_yolo_label(label)

        # Empty label = exclude
        if not lines:
            continue

        converted = validate_yolo_lines(lines)

        # Invalid label = exclude
        if converted is None:
            continue

        if not converted:
            continue

        records.append(
            {
                "image": image,
                "label": converted,
            }
        )

    return records


# ============================================================
# COLLECT UAV DATA
# ============================================================

def voc_to_yolo(xml_path):

    try:
        root = ET.parse(xml_path).getroot()

    except Exception as e:
        print(f"XML parse error: {xml_path}")
        print(e)
        return None

    size = root.find("size")

    if size is None:
        return None

    try:

        width = float(size.findtext("width"))
        height = float(size.findtext("height"))

    except Exception:
        return None

    if width <= 0 or height <= 0:
        return None

    output = []

    for obj in root.findall("object"):

        name = obj.findtext("name")

        if not name:
            continue

        name = name.strip().lower()

        # Confirmed dataset class
        if name != "uav":
            continue

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        try:

            xmin = float(bbox.findtext("xmin"))
            ymin = float(bbox.findtext("ymin"))
            xmax = float(bbox.findtext("xmax"))
            ymax = float(bbox.findtext("ymax"))

        except Exception:
            continue

        if xmax <= xmin or ymax <= ymin:
            continue

        # Clamp coordinates to image boundaries
        xmin = max(0, min(xmin, width))
        xmax = max(0, min(xmax, width))
        ymin = max(0, min(ymin, height))
        ymax = max(0, min(ymax, height))

        box_w = xmax - xmin
        box_h = ymax - ymin

        if box_w <= 0 or box_h <= 0:
            continue

        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height

        norm_w = box_w / width
        norm_h = box_h / height

        # Final class 0 = drone
        output.append(
            f"0 {x_center:.6f} {y_center:.6f} "
            f"{norm_w:.6f} {norm_h:.6f}"
        )

    return output


def collect_uav():

    records = []

    images = sorted(
        [
            p for p in UAV_IMAGE_ROOT.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
        ],
        key=lambda p: p.name.lower()
    )

    for image in images:

        xml = UAV_XML_ROOT / f"{image.stem}.xml"

        # No XML = exclude
        if not xml.exists():
            continue

        label = voc_to_yolo(xml)

        if label is None:
            continue

        if not label:
            continue

        records.append(
            {
                "image": image,
                "label": label,
            }
        )

    return records


# ============================================================
# WRITE DATA
# ============================================================

def write_records(records, split, prefix):

    for record in records:

        copy_pair(
            image=record["image"],
            label_text="\n".join(record["label"]) + "\n",
            split=split,
            prefix=prefix
        )


# ============================================================
# DATASET YAML
# ============================================================

def write_yaml():

    yaml = f"""path: {OUT.resolve().as_posix()}

train: images/train
val: images/val
test: images/test

names:
  0: drone
  1: bird
"""

    (OUT / "data.yaml").write_text(
        yaml,
        encoding="utf-8"
    )


# ============================================================
# STATISTICS
# ============================================================

def calculate_stats():

    print()
    print("=" * 100)
    print("FINAL DATASET STATISTICS")
    print("=" * 100)

    total_images = 0
    total_boxes = 0

    for split in ["train", "val", "test"]:

        image_dir = OUT / "images" / split
        label_dir = OUT / "labels" / split

        images = [
            p for p in image_dir.iterdir()
            if p.is_file()
        ]

        labels = [
            p for p in label_dir.glob("*.txt")
        ]

        boxes = 0
        classes = Counter()

        for label in labels:

            lines = [
                line.strip()
                for line in label.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

            for line in lines:

                parts = line.split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])

                classes[class_id] += 1
                boxes += 1

        print()
        print(f"{split.upper()}")
        print("-" * 60)
        print(f"Images : {len(images)}")
        print(f"Labels : {len(labels)}")
        print(f"Boxes  : {boxes}")
        print(f"Drone  : {classes[0]}")
        print(f"Bird   : {classes[1]}")

        total_images += len(images)
        total_boxes += boxes

    print()
    print("-" * 60)
    print(f"TOTAL IMAGES : {total_images}")
    print(f"TOTAL BOXES  : {total_boxes}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("BUILDING CLEAN DRONE + BIRD YOLO DATASET")
    print("=" * 100)

    print()
    print(f"RAW : {ROOT.resolve()}")
    print(f"OUT : {OUT.resolve()}")

    # -------------------------
    # Reset output
    # -------------------------

    reset_output()

    # -------------------------
    # Bird
    # -------------------------

    print()
    print("=" * 100)
    print("COLLECTING BIRD DATA")
    print("=" * 100)

    bird_train = collect_bird_split("train")
    bird_val = collect_bird_split("val")

    print(f"Valid bird train images : {len(bird_train)}")
    print(f"Valid bird val images   : {len(bird_val)}")

    # Split bird train into train/test
    random.Random(SEED).shuffle(bird_train)

    bird_train_count = int(
        len(bird_train) * BIRD_TRAIN
    )

    bird_train_final = bird_train[:bird_train_count]
    bird_test_final = bird_train[bird_train_count:]

    print()
    print(f"Bird -> train : {len(bird_train_final)}")
    print(f"Bird -> val   : {len(bird_val)}")
    print(f"Bird -> test  : {len(bird_test_final)}")

    write_records(
        bird_train_final,
        "train",
        "bird"
    )

    write_records(
        bird_val,
        "val",
        "bird"
    )

    write_records(
        bird_test_final,
        "test",
        "bird"
    )

    # -------------------------
    # UAV
    # -------------------------

    print()
    print("=" * 100)
    print("COLLECTING UAV DATA")
    print("=" * 100)

    uav_records = collect_uav()

    print(f"Valid UAV image/XML pairs : {len(uav_records)}")

    random.Random(SEED).shuffle(uav_records)

    n = len(uav_records)

    train_end = int(n * UAV_TRAIN)
    val_end = train_end + int(n * UAV_VAL)

    uav_train = uav_records[:train_end]
    uav_val = uav_records[train_end:val_end]
    uav_test = uav_records[val_end:]

    print()
    print(f"UAV -> train : {len(uav_train)}")
    print(f"UAV -> val   : {len(uav_val)}")
    print(f"UAV -> test  : {len(uav_test)}")

    write_records(
        uav_train,
        "train",
        "drone"
    )

    write_records(
        uav_val,
        "val",
        "drone"
    )

    write_records(
        uav_test,
        "test",
        "drone"
    )

    # -------------------------
    # YAML
    # -------------------------

    write_yaml()

    # -------------------------
    # Stats
    # -------------------------

    calculate_stats()

    print()
    print("=" * 100)
    print("DATASET BUILD COMPLETE")
    print("=" * 100)

    print()
    print(f"Dataset: {OUT.resolve()}")
    print(f"YAML   : {(OUT / 'data.yaml').resolve()}")

    print()
    print("FINAL CLASSES")
    print("0 = drone")
    print("1 = bird")


if __name__ == "__main__":
    main()