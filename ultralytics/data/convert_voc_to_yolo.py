import os
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse

# 默认类别列表（根据 NEU-DET 数据集设置）
DEFAULT_CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
class2id = {cls_name: idx for idx, cls_name in enumerate(DEFAULT_CLASSES)}

def convert(xml_file, output_txt_path, class_map):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    lines = []
    for obj in root.iter("object"):
        cls = obj.find("name").text.lower().strip()
        if cls not in class_map:
            continue
        cls_id = class_map[cls]
        xmlbox = obj.find("bndbox")
        xmin = int(xmlbox.find("xmin").text)
        ymin = int(xmlbox.find("ymin").text)
        xmax = int(xmlbox.find("xmax").text)
        ymax = int(xmlbox.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        bbox_w = (xmax - xmin) / w
        bbox_h = (ymax - ymin) / h
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}")

    with open(output_txt_path, "w") as f:
        f.write("\n".join(lines))


def convert_folder(xml_folder, output_folder, class_map):
    os.makedirs(output_folder, exist_ok=True)
    xml_files = [f for f in os.listdir(xml_folder) if f.endswith(".xml")]
    for xml_file in xml_files:
        xml_path = os.path.join(xml_folder, xml_file)
        output_txt_path = os.path.join(output_folder, xml_file.replace(".xml", ".txt"))
        convert(xml_path, output_txt_path, class_map)
    print(f"✅ Successfully converted {len(xml_files)} files to YOLO format at: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_dir", type=str, required=True, help="Path to XML annotation folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save YOLO label txt files")
    parser.add_argument("--class_list", nargs="+", default=DEFAULT_CLASSES, help="List of class names")

    args = parser.parse_args()
    class_map = {cls: idx for idx, cls in enumerate(args.class_list)}

    convert_folder(args.xml_dir, args.output_dir, class_map)
