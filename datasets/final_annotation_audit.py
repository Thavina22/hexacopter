from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

ROOT = Path(r".\datasets\raw")

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}


def rel(path):
    return str(path.relative_to(ROOT))


def find_images(folder):
    return sorted(
        [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ],
        key=lambda p: str(p).lower()
    )


def read_text(path):
    try:
        return path.read_text(
            encoding="utf-8",
            errors="ignore"
        )
    except Exception:
        return ""


print("=" * 100)
print("FINAL ANNOTATION AUDIT")
print("=" * 100)
print(f"ROOT: {ROOT.resolve()}")


# ============================================================
# BIRD DATASET
# ============================================================

bird_root = (
    ROOT
    / "bird-detection-yolo-dataset-main"
    / "bird-detection-yolo-dataset-main"
)

print()
print("=" * 100)
print("BIRD DATASET")
print("=" * 100)

print(f"Root: {bird_root}")


# -------------------------
# Find YAML
# -------------------------

yaml_files = sorted(
    list(bird_root.rglob("*.yaml")) +
    list(bird_root.rglob("*.yml"))
)

print()
print("YAML FILES")
print("-" * 100)

if yaml_files:
    for y in yaml_files:
        print(rel(y))
        print()
        print(read_text(y))
else:
    print("NONE FOUND")


# -------------------------
# classes.txt
# -------------------------

classes_file = bird_root / "classes.txt"

print()
print("CLASSES.TXT")
print("-" * 100)

if classes_file.exists():
    print(read_text(classes_file))
else:
    print("NOT FOUND")


# -------------------------
# Train / val
# -------------------------

bird_images = []

for split in ["train", "val"]:

    image_dir = bird_root / "images" / split
    label_dir = bird_root / "labels" / split

    print()
    print(f"BIRD SPLIT: {split}")
    print("-" * 100)

    if not image_dir.exists():
        print("IMAGE DIRECTORY NOT FOUND")
        continue

    if not label_dir.exists():
        print("LABEL DIRECTORY NOT FOUND")
        continue

    images = find_images(image_dir)

    labels = sorted(
        label_dir.glob("*.txt"),
        key=lambda p: p.name.lower()
    )

    print(f"Images : {len(images)}")
    print(f"Labels : {len(labels)}")

    image_map = {
        p.stem.lower(): p
        for p in images
    }

    label_map = {
        p.stem.lower(): p
        for p in labels
    }

    missing_labels = sorted(
        set(image_map) - set(label_map)
    )

    orphan_labels = sorted(
        set(label_map) - set(image_map)
    )

    print(f"Images without label : {len(missing_labels)}")
    print(f"Labels without image : {len(orphan_labels)}")

    if missing_labels:
        print()
        print("IMAGES WITHOUT LABEL:")
        for stem in missing_labels:
            print(rel(image_map[stem]))

    if orphan_labels:
        print()
        print("LABELS WITHOUT IMAGE:")
        for stem in orphan_labels:
            print(rel(label_map[stem]))

    # Analyze actual labels only
    total_boxes = 0
    empty_labels = []
    invalid_lines = []
    classes = Counter()

    for label in labels:

        lines = [
            line.strip()
            for line in read_text(label).splitlines()
            if line.strip()
        ]

        if not lines:
            empty_labels.append(label)
            continue

        for line_no, line in enumerate(lines, start=1):

            parts = line.split()

            if len(parts) != 5:

                invalid_lines.append(
                    (label, line_no, line)
                )

                continue

            try:

                class_id = int(float(parts[0]))

                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

            except ValueError:

                invalid_lines.append(
                    (label, line_no, line)
                )

                continue

            if not (
                0 <= x <= 1 and
                0 <= y <= 1 and
                0 < w <= 1 and
                0 < h <= 1
            ):

                invalid_lines.append(
                    (label, line_no, line)
                )

                continue

            classes[class_id] += 1
            total_boxes += 1

    print()
    print("ANNOTATION QUALITY")
    print(f"Total boxes     : {total_boxes}")
    print(f"Empty labels    : {len(empty_labels)}")
    print(f"Invalid boxes   : {len(invalid_lines)}")

    print()
    print("CLASS COUNTS")

    for class_id, count in sorted(classes.items()):
        print(f"Class {class_id}: {count}")

    if empty_labels:

        print()
        print("EMPTY LABEL FILES:")

        for label in empty_labels:
            print(rel(label))

    if invalid_lines:

        print()
        print("ACTUAL INVALID YOLO LINES:")

        for label, line_no, line in invalid_lines:
            print(
                f"{rel(label)} | "
                f"line {line_no} | {line}"
            )

    bird_images.extend(images)


# -------------------------
# Bird samples
# -------------------------

samples_dir = bird_root / "samples"

print()
print("BIRD SAMPLE IMAGES")
print("-" * 100)

if samples_dir.exists():

    samples = find_images(samples_dir)

    print(f"Sample images: {len(samples)}")

    for image in samples:
        print(rel(image))

else:
    print("NO SAMPLE DIRECTORY")


# ============================================================
# UAV DATASET
# ============================================================

uav_root = ROOT / "uav-20261001T061724Z-1-001" / "uav"

print()
print()
print("=" * 100)
print("UAV DATASET")
print("=" * 100)

print(f"Root: {uav_root}")


image_dir = uav_root / "JPEGImages"
xml_dir = uav_root / "Annotations"

images = find_images(image_dir)

xml_files = sorted(
    xml_dir.glob("*.xml"),
    key=lambda p: p.name.lower()
)

print()
print("TOTALS")
print("-" * 100)
print(f"Images : {len(images)}")
print(f"XML    : {len(xml_files)}")


# Exact filename/stem maps

image_map = {
    p.stem.lower(): p
    for p in images
}

xml_map = {
    p.stem.lower(): p
    for p in xml_files
}

missing_xml = sorted(
    set(image_map) - set(xml_map)
)

orphan_xml = sorted(
    set(xml_map) - set(image_map)
)

print()
print("PAIRING")
print("-" * 100)
print(f"Images with XML    : {len(set(image_map) & set(xml_map))}")
print(f"Images without XML : {len(missing_xml)}")
print(f"XML without image  : {len(orphan_xml)}")


if missing_xml:

    print()
    print("IMAGES WITHOUT XML")

    for stem in missing_xml:
        print(rel(image_map[stem]))


if orphan_xml:

    print()
    print("XML WITHOUT IMAGE")

    for stem in orphan_xml:
        print(rel(xml_map[stem]))


# -------------------------
# XML class and boxes
# -------------------------

classes = Counter()
total_boxes = 0
invalid_boxes = 0

for xml_file in xml_files:

    try:
        root = ET.parse(xml_file).getroot()

    except Exception as e:

        print()
        print("XML PARSE ERROR")
        print(rel(xml_file))
        print(e)

        continue

    for obj in root.findall("object"):

        name = obj.findtext("name")

        if not name:
            continue

        name = name.strip()

        bbox = obj.find("bndbox")

        if bbox is None:
            invalid_boxes += 1
            continue

        try:

            xmin = float(bbox.findtext("xmin"))
            ymin = float(bbox.findtext("ymin"))
            xmax = float(bbox.findtext("xmax"))
            ymax = float(bbox.findtext("ymax"))

        except Exception:

            invalid_boxes += 1
            continue

        if xmax <= xmin or ymax <= ymin:
            invalid_boxes += 1
            continue

        classes[name] += 1
        total_boxes += 1


print()
print("VOC ANNOTATIONS")
print("-" * 100)
print(f"Valid boxes   : {total_boxes}")
print(f"Invalid boxes : {invalid_boxes}")

print()
print("CLASSES")

for name, count in sorted(classes.items()):
    print(f"{name:<30} : {count}")


print()
print("=" * 100)
print("FINAL AUDIT COMPLETE")
print("=" * 100)