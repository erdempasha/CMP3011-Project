import os
import random
import xml.etree.ElementTree as ET
from shutil import copy2
from pathlib import Path
import yaml

# Configuration
VOC_ANNOTATIONS_DIR = "dataset/annotations"
VOC_IMAGES_DIR      = "dataset/images"
OUTPUT_DIR          = "processed/"

VALIDATION_SPLIT = 0.2

YOLO_CLASSES = [
    "with_mask",
    "without_mask",
    "mask_weared_incorrect"
]


labels_dir = os.path.join(OUTPUT_DIR, "labels")
images_dir = os.path.join(OUTPUT_DIR, "images")
os.makedirs(labels_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)


def parse_voc_annotation(xml_file):
    """Parse a single PASCAL VOC XML annotation file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_info = {
        "filename": root.find("filename").text,
        "width": int(root.find("size/width").text),
        "height": int(root.find("size/height").text),
        "objects": []
    }

    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in YOLO_CLASSES:
            continue
        class_id = YOLO_CLASSES.index(class_name)

        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        x_center = (xmin + xmax) / 2 / image_info["width"]
        y_center = (ymin + ymax) / 2 / image_info["height"]
        width = (xmax - xmin) / image_info["width"]
        height = (ymax - ymin) / image_info["height"]

        image_info["objects"].append((class_id, x_center, y_center, width, height))

    return image_info


def convert_annotations():
    all_files = list(Path(VOC_ANNOTATIONS_DIR).glob("*.xml"))
    random.shuffle(all_files)

    val_count = int(len(all_files) * VALIDATION_SPLIT)
    val_files = set(all_files[:val_count])

    for xml_file in all_files:
        image_info = parse_voc_annotation(xml_file)
        image_file = os.path.join(VOC_IMAGES_DIR, image_info["filename"])

        if not os.path.exists(image_file):
            print(f"Warning: Image file {image_file} not found.")
            continue

        split_folder = "val" if xml_file in val_files else "train"
        dest_label_folder = os.path.join(labels_dir, split_folder)
        os.makedirs(dest_label_folder, exist_ok=True)
        label_file = os.path.join(dest_label_folder, os.path.splitext(image_info["filename"])[0] + ".txt")
        with open(label_file, "w") as f:
            for obj in image_info["objects"]:
                f.write(f"{obj[0]} {obj[1]:.6f} {obj[2]:.6f} {obj[3]:.6f} {obj[4]:.6f}\n")

        dest_image_folder = os.path.join(images_dir, split_folder)
        os.makedirs(dest_image_folder, exist_ok=True)
        copy2(image_file, dest_image_folder)


def create_yaml_file(output_dir, class_names):

    yaml_content = {
        "path" : os.path.join("../", output_dir),
        "train": "images/train",
        "val"  : "images/val",
        "names": dict(enumerate(class_names))
    }
    
    yaml_file_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump(
            yaml_content,
            yaml_file,
            sort_keys = False
        )
    
    print(f"YAML file created at {yaml_file_path}")

if __name__ == "__main__":
    convert_annotations()
    create_yaml_file(OUTPUT_DIR, YOLO_CLASSES)
    print(f"Conversion complete. Files saved to {OUTPUT_DIR}.")
