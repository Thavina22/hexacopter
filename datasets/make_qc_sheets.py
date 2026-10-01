from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r".\datasets\raw")
OUT = Path(r".\datasets\qc")

OUT.mkdir(parents=True, exist_ok=True)

FONT = ImageFont.load_default()


def make_sheet(images, output, title, columns=4):

    if not images:
        print(f"No images for: {title}")
        return

    thumb_w = 320
    thumb_h = 220
    label_h = 30

    rows = (len(images) + columns - 1) // columns

    sheet = Image.new(
        "RGB",
        (
            columns * thumb_w,
            rows * (thumb_h + label_h)
        ),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for i, image_path in enumerate(images):

        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((thumb_w - 10, thumb_h - 10))

        except Exception as e:
            print(f"Could not open: {image_path}")
            print(e)
            continue

        x = (i % columns) * thumb_w
        y = (i // columns) * (thumb_h + label_h)

        px = x + (thumb_w - image.width) // 2
        py = y + 5

        sheet.paste(image, (px, py))

        name = image_path.name

        draw.text(
            (x + 5, y + thumb_h),
            name,
            fill="black",
            font=FONT
        )

    sheet.save(output)

    print(f"Created: {output}")


# ============================================================
# BIRD EMPTY LABELS
# ============================================================

bird_root = (
    ROOT
    / "bird-detection-yolo-dataset-main"
    / "bird-detection-yolo-dataset-main"
)

empty_bird_ids = [
    "1762",
    "1852",
    "1884",
    "1992",
    "2160",
    "2531",
    "2533",
    "372",
    "457",
    "534",
    "581",
    "2056",
]

bird_images = []

for image_dir in [
    bird_root / "images" / "train",
    bird_root / "images" / "val",
]:

    for image_id in empty_bird_ids:

        for ext in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        ]:

            candidate = image_dir / f"{image_id}{ext}"

            if candidate.exists():
                bird_images.append(candidate)

sheet = OUT / "bird_empty_labels.jpg"

make_sheet(
    bird_images,
    sheet,
    "Bird empty labels"
)


# ============================================================
# UAV IMAGES WITHOUT XML
# ============================================================

uav_root = (
    ROOT
    / "uav-20261001T061724Z-1-001"
    / "uav"
)

image_dir = uav_root / "JPEGImages"
xml_dir = uav_root / "Annotations"

uav_images = list(
    image_dir.glob("*.jpg")
)

uav_xml_stems = {
    x.stem.lower()
    for x in xml_dir.glob("*.xml")
}

unmatched_uav = [
    image
    for image in uav_images
    if image.stem.lower() not in uav_xml_stems
]

sheet = OUT / "uav_images_without_xml.jpg"

make_sheet(
    unmatched_uav,
    sheet,
    "UAV images without XML"
)


print()
print("=" * 80)
print("QC SHEETS CREATED")
print("=" * 80)
print(f"Bird empty-label images : {len(bird_images)}")
print(f"UAV images without XML  : {len(unmatched_uav)}")
print(f"Output directory        : {OUT.resolve()}")