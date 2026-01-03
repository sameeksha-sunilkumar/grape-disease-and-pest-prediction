import os
from PIL import Image

DATA_DIR = r"C:\Users\DELL\Desktop\projects\grape disease detection\fruit_data"

removed = 0
checked = 0

for split in ["train", "test"]:
    split_path = os.path.join(DATA_DIR, split)
    for cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, cls)
        if not os.path.isdir(cls_path):
            continue

        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            checked += 1
            try:
                with Image.open(img_path) as img:
                    img.verify()   
            except Exception:
                print(f"Removing corrupted image: {img_path}")
                os.remove(img_path)
                removed += 1

print(f"\nChecked {checked} images")
print(f" Removed {removed} corrupted images")
