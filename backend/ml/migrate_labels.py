"""
One-time script: copies existing labels from _LABEL_ROOT tree
into their correct location beside each image.

Run ONCE inside the container before --stage train.
After this, future --stage annotate runs will write labels
beside images directly (fixed in train_yolo_finetuned.py).
"""
from pathlib import Path
import shutil

LABEL_ROOT = Path("/app/data/datasets/interior_design_images_metadata/labels")
IMAGE_ROOTS = [
    Path("/app/data/datasets/interior_design_images_metadata"),
    Path("/app/data/datasets/interior_design_material_style"),
]
ROOM_LABELS  = ["bathroom", "bedroom", "kitchen", "living_room"]
STYLE_LABELS = ["boho", "industrial", "minimalist", "modern", "scandinavian"]

copied = 0
missing_image = 0
already_exists = 0

for room in ROOM_LABELS:
    for style in STYLE_LABELS:
        label_dir = LABEL_ROOT / room / style
        if not label_dir.is_dir():
            continue
        for label_file in label_dir.glob("*.txt"):
            stem = label_file.stem  # e.g. bathroom_boho_26
            # Find the matching image in any image root
            found_img = None
            for img_root in IMAGE_ROOTS:
                for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
                    candidate = img_root / room / style / (stem + ext)
                    if candidate.exists():
                        found_img = candidate
                        break
                if found_img:
                    break

            if found_img is None:
                missing_image += 1
                continue

            dest = found_img.with_suffix(".txt")
            if dest.exists():
                already_exists += 1
                continue

            shutil.copy2(label_file, dest)
            copied += 1

print(f"Done.")
print(f"  Labels copied beside images : {copied}")
print(f"  Already existed (skipped)   : {already_exists}")
print(f"  No matching image found     : {missing_image}")
print(f"\nNow run: python ml/train_yolo_finetuned.py --stage train --weights-dir ml/weights --epochs 50")
