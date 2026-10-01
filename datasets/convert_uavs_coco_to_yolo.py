import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\thavi\Downloads\Drone_Bird_Detection")

SOURCE_ROOT = PROJECT_ROOT / "datasets" / "extracted" / "UAVs"
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "yolo" / "UAVs"

SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

def convert_split(source_split, output_split):
    source_dir = SOURCE_ROOT / source_split
    output_images = OUTPUT_ROOT / "images" / output_split
    output_labels = OUTPUT_ROOT / "labels" / output_split

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    json_path = source_dir / "_annotations.coco.json"

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Only category 1 = drone
    image_info = {
        image["id"]: image
        for image in coco["images"]
    }

    annotations_by_image = {}

    for ann in coco["annotations"]:
        if ann["category_id"] != 1:
            continue

        annotations_by_image.setdefault(
            ann["image_id"], []
        ).append(ann)

    converted = 0
    missing_labels = 0

    for image_id, image in image_info.items():

        filename = image["file_name"]
        source_image = source_dir / filename

        if not source_image.exists():
            print(f"WARNING: Missing image: {source_image}")
            continue

        # Copy image
        shutil.copy2(
            source_image,
            output_images / filename
        )

        width = image["width"]
        height = image["height"]

        label_path = output_labels / (
            Path(filename).stem + ".txt"
        )

        anns = annotations_by_image.get(image_id, [])

        lines = []

        for ann in anns:

            x, y, w, h = ann["bbox"]

            # COCO -> YOLO
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height

            w_norm = w / width
            h_norm = h / height

            # YOLO class 0 = drone
            lines.append(
                f"0 {x_center:.6f} {y_center:.6f} "
                f"{w_norm:.6f} {h_norm:.6f}"
            )

        if lines:
            label_path.write_text(
                "\n".join(lines),
                encoding="utf-8"
            )
            converted += 1
        else:
            # Keep an empty label for an image without annotations
            label_path.write_text("", encoding="utf-8")
            missing_labels += 1

    print(f"\n{source_split} -> {output_split}")
    print(f"Images copied : {len(list(output_images.glob('*')))}")
    print(f"Labels created: {converted}")
    print(f"Empty labels  : {missing_labels}")


def main():

    print("COCO → YOLO conversion")
    print("======================")

    for source_split, output_split in SPLITS.items():
        convert_split(source_split, output_split)

    print("\nConversion completed!")
    print(f"Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()