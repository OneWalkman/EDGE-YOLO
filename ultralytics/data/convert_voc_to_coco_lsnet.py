import os
import json
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

def convert_voc_to_coco(xml_list, class_list, output_json, image_dir_map):
    json_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    ann_id = 1

    for idx, cls in enumerate(class_list):
        json_dict["categories"].append({"id": idx, "name": cls, "supercategory": "defect"})

    for img_id, xml_path in enumerate(tqdm(xml_list)):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename").text
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        json_dict["images"].append({
            "file_name": filename if Path(filename).suffix else f"{filename}.jpg",
            "height": height,
            "width": width,
            "id": img_id
        })

        for obj in root.findall("object"):
            cls_name = obj.find("name").text.strip().lower()
            if cls_name not in class_list:
                continue
            cls_id = class_list.index(cls_name)
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            width_box = xmax - xmin
            height_box = ymax - ymin

            json_dict["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": [xmin, ymin, width_box, height_box],
                "area": width_box * height_box,
                "iscrowd": 0,
                "segmentation": []
            })
            ann_id += 1

    with open(output_json, 'w') as f:
        json.dump(json_dict, f, indent=2)
    print(f"✅ COCO JSON saved to {output_json}")


def find_image_file(img_dir, filename_stem):
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        candidate = img_dir / f"{filename_stem}{ext}"
        if candidate.exists():
            return candidate
    return None

def prepare_lsnet_dataset(voc_root, output_root, val_ratio=0.1):
    voc_root = Path(voc_root)
    output_root = Path(output_root)
    xml_dir = voc_root / "ANNOTATIONS"
    img_dir = voc_root / "IMAGES"

    output_train_img_dir = output_root / "coco/train2017"
    output_val_img_dir = output_root / "coco/val2017"
    output_anno_dir = output_root / "coco/annotations"
    output_train_img_dir.mkdir(parents=True, exist_ok=True)
    output_val_img_dir.mkdir(parents=True, exist_ok=True)
    output_anno_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(list(xml_dir.glob("*.xml")))
    train_xmls, val_xmls = train_test_split(xml_files, test_size=val_ratio, random_state=42)

    train_map = {}
    for xml in tqdm(train_xmls, desc="Copying train images"):
        filename = ET.parse(xml).getroot().find("filename").text
        stem = Path(filename).stem
        src_img = find_image_file(img_dir, stem)
        if not src_img:
            print(f"❌ Image not found for: {stem} (from XML: {xml.name})")
            continue
        shutil.copy(src_img, output_train_img_dir / src_img.name)
        train_map[xml] = img_dir

    val_map = {}
    for xml in tqdm(val_xmls, desc="Copying val images"):
        filename = ET.parse(xml).getroot().find("filename").text
        stem = Path(filename).stem
        src_img = find_image_file(img_dir, stem)
        if not src_img:
            print(f"❌ Image not found for: {stem} (from XML: {xml.name})")
            continue
        shutil.copy(src_img, output_val_img_dir / src_img.name)
        val_map[xml] = img_dir

    classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    convert_voc_to_coco(train_map.keys(), classes, output_anno_dir / "instances_train2017.json", train_map)
    convert_voc_to_coco(val_map.keys(), classes, output_anno_dir / "instances_val2017.json", val_map)

if __name__ == "__main__":
    prepare_lsnet_dataset(
        voc_root=r"D:\KaggleData\NEU-DET",
        output_root=r"D:\KaggleData\NEU-DET-LSNet"
    )
