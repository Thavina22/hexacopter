from pathlib import Path
from collections import Counter

root = Path(r".\datasets\extracted\cloud_drone_aug")

images = list(root.rglob("*.jpg"))
labels = list(root.rglob("*.txt"))

boxes = 0
empty = 0
invalid = 0
classes = Counter()
bad_examples = []

for label in labels:

    lines = [
        line.strip()
        for line in label.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()
        if line.strip()
    ]

    if not lines:
        empty += 1
        continue

    for line in lines:

        parts = line.split()

        if len(parts) != 5:
            invalid += 1

            if len(bad_examples) < 10:
                bad_examples.append(
                    (label.name, line)
                )

            continue

        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])

        except ValueError:

            invalid += 1

            if len(bad_examples) < 10:
                bad_examples.append(
                    (label.name, line)
                )

            continue

        if not (
            0 <= x <= 1
            and 0 <= y <= 1
            and 0 < w <= 1
            and 0 < h <= 1
        ):

            invalid += 1

            if len(bad_examples) < 10:
                bad_examples.append(
                    (label.name, line)
                )

            continue

        classes[cls] += 1
        boxes += 1


print("=" * 80)
print("CLOUD DRONE AUG AUDIT")
print("=" * 80)

print(f"Images          : {len(images)}")
print(f"TXT labels      : {len(labels)}")
print(f"Boxes           : {boxes}")
print(f"Empty labels    : {empty}")
print(f"Invalid boxes   : {invalid}")
print(f"Classes         : {dict(classes)}")

print()
print("BAD EXAMPLES")
print("-" * 80)

for filename, line in bad_examples:
    print(filename)
    print(" ", line)

print()
print("=" * 80)