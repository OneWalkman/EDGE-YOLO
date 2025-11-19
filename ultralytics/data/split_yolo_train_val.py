
import os
import random
import shutil
from pathlib import Path

def split_yolo_dataset(root_dir, val_ratio=0.1):
    root = Path(root_dir)
    images_dir = root / "IMAGES"
    labels_dir = root / "YOLO_LABELS"

    output_dir = root / "YOLO_SPLIT"
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 获取所有图像和对应的标签
    all_images = sorted([f for f in images_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    print(f"Found {len(all_images)} images.")

    # 打乱并划分
    random.seed(42)
    random.shuffle(all_images)
    val_count = int(len(all_images) * val_ratio)
    val_images = all_images[:val_count]
    train_images = all_images[val_count:]

    def copy_files(image_list, target_img_dir, target_lbl_dir):
        for img_path in image_list:
            shutil.copy(img_path, target_img_dir / img_path.name)
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy(label_path, target_lbl_dir / label_path.name)
            else:
                print(f"⚠️ Label not found for image: {img_path.name}")

    print("📦 Copying training set...")
    copy_files(train_images, train_img_dir, train_lbl_dir)
    print("📦 Copying validation set...")
    copy_files(val_images, val_img_dir, val_lbl_dir)
    print(f"✅ Done. Split result saved in: {output_dir}")

if __name__ == "__main__":
    split_yolo_dataset(r"D:\KaggleData\NEU-DET", val_ratio=0.1)
