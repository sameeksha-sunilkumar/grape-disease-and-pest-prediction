import os
import shutil
import random

SOURCE_DIR = "dataset"
TRAIN_DIR = "fruit_data/train"
TEST_DIR  = "fruit_data/test"

split_ratio = 0.8  

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

for cls in os.listdir(SOURCE_DIR):
    cls_path = os.path.join(SOURCE_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    random.shuffle(images)

    split = int(len(images) * split_ratio)
    train_imgs = images[:split]
    test_imgs  = images[split:]

    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(TRAIN_DIR, cls, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(TEST_DIR, cls, img)
        )

    print(f"{cls}: {len(train_imgs)} train, {len(test_imgs)} test")

print("Fruit dataset split completed")
