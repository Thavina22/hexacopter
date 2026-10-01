from pathlib import Path
from collections import Counter, defaultdict
import json
import xml.etree.ElementTree as ET

try:
    from PIL import Image
except ImportError:
    print("Pillow not installed.")
    print("Run: pip install pillow")
    raise


PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "datasets" / "raw"


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTS = {".txt", ".xml", ".json"}


def is_image(path):
    return path.suffix.lower() in IMAGE_EXTS


def is_annotation(path):
    return path.suffix.lower() in LABEL_EXTS


def image_info(images):
    resolutions = Counter()
    corrupt = []

    for img_path in images:
        try:
            with Image.open(img_path) as im:
                resolutions[im.size] += 1
        except Exception:
            corrupt.append(str(img_path))

    return resolutions, corrupt


def audit_yolo_txt(dataset_root, label_files):
    """
    Find YOLO TXT labels and count class IDs / boxes.
    """
    class_counts = Counter()
    total_boxes = 0
    empty_labels = 0
    invalid_boxes = 0
    valid_labels = 0

    for label in label_files:
        try:
            text = label.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue

        if not text:
            empty_labels += 1
            continue

        file_valid = True

        for line in text.splitlines():
            parts = line.split()

            if len(parts) != 5:
                invalid_boxes += 1
                file_valid = False
                continue

            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])

                if not (0 <= x <= 1 and
                        0 <= y <= 1 and
                        0 < w <= 1 and
                        0 < h <= 1):
                    invalid_boxes += 1
                    file_valid = False
                    continue

                class_counts[cls] += 1
                total_boxes += 1

            except Exception:
                invalid_boxes += 1
                file_valid = False

        if file_valid:
            valid_labels += 1

    return {
        "class_counts": class_counts,
        "total_boxes": total_boxes,
        "empty_labels": empty_labels,
        "invalid_boxes": invalid_boxes,
        "valid_labels": valid_labels,
    }


def audit_voc_xml(xml_files):
    class_counts = Counter()
    total_boxes = 0
    invalid_boxes = 0

    for xml in xml_files:
        try:
            root = ET.parse(xml).getroot()

            for obj in root.findall("object"):
                name = obj.findtext("name")

                if not name:
                    invalid_boxes += 1
                    continue

                class_counts[name.strip()] += 1
                total_boxes += 1

        except Exception:
            invalid_boxes += 1

    return {
        "class_counts": class_counts,
        "total_boxes": total_boxes,
        "invalid_boxes": invalid_boxes,
    }


def audit_coco_json(json_files):
    results = []

    for js in json_files:
        try:
            data = json.loads(js.read_text(
                encoding="utf-8",
                errors="ignore"
            ))

            categories = {
                c["id"]: c.get("name", f"class_{c['id']}")
                for c in data.get("categories", [])
            }

            annotations = data.get("annotations", [])

            counts = Counter()

            for ann in annotations:
                cat_id = ann.get("category_id")
                counts[categories.get(cat_id, f"class_{cat_id}")] += 1

            results.append({
                "file": js.name,
                "images": len(data.get("images", [])),
                "annotations": len(annotations),
                "classes": categories,
                "class_counts": counts,
            })

        except Exception as e:
            results.append({
                "file": js.name,
                "error": str(e)
            })

    return results


def find_dataset_roots():
    """
    Treat every direct folder under raw/ as a dataset.
    """
    return sorted(
        p for p in RAW.iterdir()
        if p.is_dir()
    )


def audit_dataset(dataset):
    print("\n" + "=" * 80)
    print(f"DATASET: {dataset.name}")
    print("=" * 80)

    all_files = [
        p for p in dataset.rglob("*")
        if p.is_file()
    ]

    images = [p for p in all_files if is_image(p)]

    txt_files = [
        p for p in all_files
        if p.suffix.lower() == ".txt"
    ]

    xml_files = [
        p for p in all_files
        if p.suffix.lower() == ".xml"
    ]

    json_files = [
        p for p in all_files
        if p.suffix.lower() == ".json"
    ]

    yaml_files = [
        p for p in all_files
        if p.suffix.lower() in {".yaml", ".yml"}
    ]

    print(f"Files                : {len(all_files):,}")
    print(f"Images               : {len(images):,}")
    print(f"TXT files            : {len(txt_files):,}")
    print(f"XML files            : {len(xml_files):,}")
    print(f"JSON files           : {len(json_files):,}")
    print(f"YAML files           : {len(yaml_files):,}")

    # ---------------------------------------------------------
    # Image information
    # ---------------------------------------------------------

    resolutions, corrupt = image_info(images)

    print("\nIMAGE RESOLUTIONS")
    print("-" * 80)

    for resolution, count in resolutions.most_common(15):
        print(f"{resolution[0]} x {resolution[1]} : {count:,}")

    if corrupt:
        print(f"\nCORRUPT IMAGES       : {len(corrupt):,}")
    else:
        print("\nCORRUPT IMAGES       : 0")

    # ---------------------------------------------------------
    # YOLO TXT
    # ---------------------------------------------------------

    if txt_files:
        result = audit_yolo_txt(dataset, txt_files)

        print("\nYOLO TXT ANALYSIS")
        print("-" * 80)

        print(f"Total boxes          : {result['total_boxes']:,}")
        print(f"Empty label files    : {result['empty_labels']:,}")
        print(f"Invalid boxes        : {result['invalid_boxes']:,}")
        print(f"Valid label files    : {result['valid_labels']:,}")

        print("\nCLASS IDS")
        for cls, count in sorted(result["class_counts"].items()):
            print(f"Class {cls}              : {count:,}")

    # ---------------------------------------------------------
    # VOC XML
    # ---------------------------------------------------------

    if xml_files:
        result = audit_voc_xml(xml_files)

        print("\nVOC XML ANALYSIS")
        print("-" * 80)

        print(f"Total boxes          : {result['total_boxes']:,}")
        print(f"Invalid boxes        : {result['invalid_boxes']:,}")

        print("\nCLASSES")

        for cls, count in result["class_counts"].most_common():
            print(f"{cls:25s}: {count:,}")

    # ---------------------------------------------------------
    # COCO JSON
    # ---------------------------------------------------------

    if json_files:
        results = audit_coco_json(json_files)

        print("\nCOCO JSON ANALYSIS")
        print("-" * 80)

        for result in results:
            print(f"\nFile: {result['file']}")

            if "error" in result:
                print(f"ERROR: {result['error']}")
                continue

            print(f"Images             : {result['images']:,}")
            print(f"Annotations        : {result['annotations']:,}")

            print("\nCategories")

            for cat_id, name in result["classes"].items():
                count = result["class_counts"].get(name, 0)
                print(f"{cat_id}: {name:25s} -> {count:,}")

    # ---------------------------------------------------------
    # YAML
    

    if not RAW.exists():
        print(f"ERROR: raw dataset directory does not exist:")
        print(RAW)
        return

    datasets = find_dataset_roots()

    if not datasets:
        print("No dataset folders found.")
        return

    print("=" * 80)
    print("DRONE + BIRD DATASET AUDIT")
    print("=" * 80)
    print(f"Project : {PROJECT}")
    print(f"Raw     : {RAW}")
    print(f"Datasets: {len(datasets)}")

    for dataset in datasets:
        audit_dataset(dataset)

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()