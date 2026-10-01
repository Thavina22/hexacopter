from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r".\datasets\raw")

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

LABEL_EXTS = {
    ".txt",
    ".xml",
    ".json",
}


# ============================================================
# HELPERS
# ============================================================

def is_image(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def relative(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def inspect_folder(folder):
    files = [p for p in folder.iterdir() if p.is_file()]

    images = [p for p in files if is_image(p)]

    txt = [p for p in files if p.suffix.lower() == ".txt"]
    xml = [p for p in files if p.suffix.lower() == ".xml"]
    json_files = [p for p in files if p.suffix.lower() == ".json"]

    return {
        "images": images,
        "txt": txt,
        "xml": xml,
        "json": json_files,
    }


# ============================================================
# IMAGE INVENTORY
# ============================================================

def print_image_inventory(dataset):

    print()
    print("IMAGE LOCATIONS")
    print("-" * 100)

    image_folders = []

    for folder in sorted(
        [dataset] + [p for p in dataset.rglob("*") if p.is_dir()],
        key=lambda p: str(p).lower()
    ):

        images = [
            p for p in folder.iterdir()
            if is_image(p)
        ]

        if images:
            image_folders.append((folder, images))

    if not image_folders:
        print("NO IMAGES FOUND")
        return

    total = 0

    for folder, images in image_folders:

        print(
            f"{relative(folder, dataset):<70}"
            f"{len(images):>8} images"
        )

        total += len(images)

    print("-" * 100)
    print(f"TOTAL IMAGES: {total}")


# ============================================================
# RECURSIVE FILE INVENTORY
# ============================================================

def print_recursive_inventory(dataset):

    print()
    print("RECURSIVE FOLDER INVENTORY")
    print("-" * 100)

    folders = sorted(
        [dataset] + [p for p in dataset.rglob("*") if p.is_dir()],
        key=lambda p: str(p).lower()
    )

    for folder in folders:

        counts = inspect_folder(folder)

        all_files = list(folder.iterdir())

        directories = [
            p for p in all_files
            if p.is_dir()
        ]

        print(
            f"{relative(folder, dataset):<65} "
            f"dirs={len(directories):<4} "
            f"images={len(counts['images']):<6} "
            f"txt={len(counts['txt']):<6} "
            f"xml={len(counts['xml']):<6} "
            f"json={len(counts['json']):<6}"
        )


# ============================================================
# YOLO LABEL INSPECTION
# ============================================================

def inspect_yolo_labels(dataset):

    label_files = sorted(
        dataset.rglob("*.txt"),
        key=lambda p: str(p).lower()
    )

    if not label_files:
        return

    print()
    print("YOLO LABEL ANALYSIS")
    print("-" * 100)

    total_boxes = 0
    empty_files = 0
    invalid_lines = 0

    classes = Counter()

    for label in label_files:

        try:
            lines = label.read_text(
                encoding="utf-8",
                errors="ignore"
            ).splitlines()

        except Exception as e:
            print(f"READ ERROR: {relative(label, dataset)}")
            print(f"           {e}")
            continue

        valid_lines = [
            line.strip()
            for line in lines
            if line.strip()
        ]

        if not valid_lines:
            empty_files += 1
            continue

        for line_number, line in enumerate(valid_lines, start=1):

            parts = line.split()

            if len(parts) != 5:
                invalid_lines += 1
                print(
                    f"INVALID: {relative(label, dataset)} "
                    f"line {line_number}: {line}"
                )
                continue

            try:
                class_id = int(float(parts[0]))

                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

            except ValueError:

                invalid_lines += 1

                print(
                    f"INVALID NUMBERS: "
                    f"{relative(label, dataset)} "
                    f"line {line_number}: {line}"
                )

                continue

            # YOLO normalized coordinate validation

            if not (
                0 <= x <= 1 and
                0 <= y <= 1 and
                0 < w <= 1 and
                0 < h <= 1
            ):

                invalid_lines += 1

                print(
                    f"INVALID BOX: "
                    f"{relative(label, dataset)} "
                    f"line {line_number}: {line}"
                )

                continue

            classes[class_id] += 1
            total_boxes += 1

    print(f"Label files       : {len(label_files)}")
    print(f"Total valid boxes : {total_boxes}")
    print(f"Empty label files : {empty_files}")
    print(f"Invalid lines     : {invalid_lines}")

    print()
    print("CLASS COUNTS")
    print("-" * 100)

    for class_id, count in sorted(classes.items()):
        print(f"Class {class_id:<5}: {count}")


# ============================================================
# VOC XML INSPECTION
# ============================================================

def inspect_voc(dataset):

    xml_files = sorted(
        dataset.rglob("*.xml"),
        key=lambda p: str(p).lower()
    )

    if not xml_files:
        return

    print()
    print("PASCAL VOC XML ANALYSIS")
    print("-" * 100)

    classes = Counter()

    total_boxes = 0
    invalid_boxes = 0

    for xml_file in xml_files:

        try:
            root = ET.parse(xml_file).getroot()

        except Exception as e:

            print(
                f"XML ERROR: {relative(xml_file, dataset)}"
            )

            print(f"           {e}")

            continue

        for obj in root.findall("object"):

            name_node = obj.find("name")

            if name_node is None:
                continue

            class_name = name_node.text.strip()

            classes[class_name] += 1

            bbox = obj.find("bndbox")

            if bbox is None:
                invalid_boxes += 1
                continue

            try:

                xmin = float(bbox.findtext("xmin"))
                ymin = float(bbox.findtext("ymin"))
                xmax = float(bbox.findtext("xmax"))
                ymax = float(bbox.findtext("ymax"))

                if (
                    xmax <= xmin or
                    ymax <= ymin
                ):
                    invalid_boxes += 1
                    continue

                total_boxes += 1

            except Exception:

                invalid_boxes += 1

    print(f"XML files        : {len(xml_files)}")
    print(f"Valid boxes      : {total_boxes}")
    print(f"Invalid boxes    : {invalid_boxes}")

    print()
    print("CLASSES")
    print("-" * 100)

    for class_name, count in sorted(classes.items()):
        print(f"{class_name:<30} : {count}")


# ============================================================
# IMAGE / LABEL MATCHING
# ============================================================

def inspect_pairs(dataset):

    images = {
        p.stem.lower(): p
        for p in dataset.rglob("*")
        if is_image(p)
    }

    txt_labels = {
        p.stem.lower(): p
        for p in dataset.rglob("*.txt")
    }

    xml_labels = {
        p.stem.lower(): p
        for p in dataset.rglob("*.xml")
    }

    print()
    print("IMAGE / ANNOTATION MATCHING")
    print("-" * 100)

    print(f"Images       : {len(images)}")
    print(f"TXT labels   : {len(txt_labels)}")
    print(f"XML labels   : {len(xml_labels)}")

    # TXT matching

    if txt_labels:

        missing_txt = []

        for stem, label in txt_labels.items():

            if stem not in images:
                missing_txt.append(label)

        images_without_txt = []

        for stem, image in images.items():

            if stem not in txt_labels:
                images_without_txt.append(image)

        print()
        print("YOLO TXT MATCHING")
        print(f"Labels without image : {len(missing_txt)}")
        print(f"Images without label : {len(images_without_txt)}")

        if missing_txt:

            print()
            print("LABELS WITHOUT IMAGE:")

            for p in missing_txt[:50]:
                print(relative(p, dataset))

        if images_without_txt:

            print()
            print("IMAGES WITHOUT LABEL:")

            for p in images_without_txt[:50]:
                print(relative(p, dataset))

    # XML matching

    if xml_labels:

        missing_xml = []

        for stem, label in xml_labels.items():

            if stem not in images:
                missing_xml.append(label)

        images_without_xml = []

        for stem, image in images.items():

            if stem not in xml_labels:
                images_without_xml.append(image)

        print()
        print("VOC XML MATCHING")
        print(f"XML without image : {len(missing_xml)}")
        print(f"Images without XML: {len(images_without_xml)}")

        if missing_xml:

            print()
            print("XML WITHOUT IMAGE:")

            for p in missing_xml[:50]:
                print(relative(p, dataset))

        if images_without_xml:

            print()
            print("IMAGES WITHOUT XML:")

            for p in images_without_xml[:50]:
                print(relative(p, dataset))


# ============================================================
# DATASET
# ============================================================

def inspect_dataset(dataset):

    print()
    print()
    print("=" * 100)
    print(f"DATASET: {dataset.name}")
    print("=" * 100)

    print_recursive_inventory(dataset)

    print_image_inventory(dataset)

    inspect_yolo_labels(dataset)

    inspect_voc(dataset)

    inspect_pairs(dataset)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("DEEP DRONE + BIRD DATASET INSPECTION")
    print("=" * 100)

    print(f"ROOT: {ROOT.resolve()}")

    if not ROOT.exists():

        print()
        print("ERROR:")
        print("Raw dataset directory does not exist.")

        return

    datasets = sorted(
        [
            p for p in ROOT.iterdir()
            if p.is_dir()
        ],
        key=lambda p: p.name.lower()
    )

    print()
    print(f"DATASETS FOUND: {len(datasets)}")

    for dataset in datasets:
        inspect_dataset(dataset)

    print()
    print()
    print("=" * 100)
    print("DEEP INSPECTION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()