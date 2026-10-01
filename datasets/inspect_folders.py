from pathlib import Path
from collections import Counter

ROOT = Path(r".\datasets\raw")

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}

ANNOTATION_EXTS = {
    ".txt", ".xml", ".json", ".yaml", ".yml"
}


def count_files(folder):
    counts = Counter()

    for item in folder.iterdir():
        if item.is_file():
            ext = item.suffix.lower()

            if ext in IMAGE_EXTS:
                counts["images"] += 1
            elif ext == ".txt":
                counts["txt"] += 1
            elif ext == ".xml":
                counts["xml"] += 1
            elif ext == ".json":
                counts["json"] += 1
            elif ext in {".yaml", ".yml"}:
                counts["yaml"] += 1
            else:
                counts["other"] += 1

    return counts


print("=" * 100)
print("COMPLETE RAW DATASET FOLDER INSPECTION")
print("=" * 100)
print(f"Root: {ROOT.resolve()}")
print()


if not ROOT.exists():
    print("ERROR: Raw dataset folder does not exist.")
    raise SystemExit(1)


datasets = sorted(
    [p for p in ROOT.iterdir() if p.is_dir()],
    key=lambda p: p.name.lower()
)


for dataset in datasets:

    print()
    print("=" * 100)
    print(f"DATASET: {dataset.name}")
    print("=" * 100)

    # ---------------------------------------------------------
    # Every directory
    # ---------------------------------------------------------

    folders = [dataset] + sorted(
        [p for p in dataset.rglob("*") if p.is_dir()],
        key=lambda p: str(p).lower()
    )

    print()
    print("FOLDERS")
    print("-" * 100)

    for folder in folders:

        counts = count_files(folder)

        relative = folder.relative_to(dataset)

        if relative == Path("."):
            display_name = "."
        else:
            display_name = str(relative)

        print(
            f"{display_name:<70} "
            f"images={counts['images']:<6} "
            f"txt={counts['txt']:<6} "
            f"xml={counts['xml']:<6} "
            f"json={counts['json']:<6} "
            f"yaml={counts['yaml']:<4}"
        )

    # ---------------------------------------------------------
    # Recursive totals
    # ---------------------------------------------------------

    all_files = [
        p for p in dataset.rglob("*")
        if p.is_file()
    ]

    images = [
        p for p in all_files
        if p.suffix.lower() in IMAGE_EXTS
    ]

    txt = [
        p for p in all_files
        if p.suffix.lower() == ".txt"
    ]

    xml = [
        p for p in all_files
        if p.suffix.lower() == ".xml"
    ]

    json_files = [
        p for p in all_files
        if p.suffix.lower() == ".json"
    ]

    yaml = [
        p for p in all_files
        if p.suffix.lower() in {".yaml", ".yml"}
    ]

    print()
    print("RECURSIVE TOTALS")
    print("-" * 100)
    print(f"Images : {len(images)}")
    print(f"TXT    : {len(txt)}")
    print(f"XML    : {len(xml)}")
    print(f"JSON   : {len(json_files)}")
    print(f"YAML   : {len(yaml)}")

    # ---------------------------------------------------------
    # Image folders
    # ---------------------------------------------------------

    print()
    print("FOLDERS CONTAINING IMAGES")
    print("-" * 100)

    image_folders = Counter(
        str(p.parent.relative_to(dataset))
        for p in images
    )

    for folder, count in sorted(image_folders.items()):
        print(f"{folder:<70} {count} images")

    # ---------------------------------------------------------
    # Annotation folders
    # ---------------------------------------------------------

    print()
    print("FOLDERS CONTAINING ANNOTATIONS")
    print("-" * 100)

    annotation_files = txt + xml + json_files

    annotation_folders = Counter(
        str(p.parent.relative_to(dataset))
        for p in annotation_files
    )

    for folder, count in sorted(annotation_folders.items()):
        print(f"{folder:<70} {count} annotation files")

    # ---------------------------------------------------------
    # YAML locations
    # ---------------------------------------------------------

    if yaml:

        print()
        print("YAML FILES")
        print("-" * 100)

        for y in yaml:
            print(y.relative_to(dataset))

    # ---------------------------------------------------------
    # Sample images
    # ---------------------------------------------------------

    print()
    print("SAMPLE IMAGE PATHS")
    print("-" * 100)

    for image in images[:10]:
        print(image.relative_to(dataset))


print()
print("=" * 100)
print("INSPECTION COMPLETE")
print("=" * 100)