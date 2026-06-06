import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


SOURCE_ROOT = Path("PCB_DATASET")
IMAGE_ROOT = SOURCE_ROOT / "Images"
ANNOTATION_ROOT = SOURCE_ROOT / "Annotations"

OUTPUT_ROOT = Path("PCB_YOLO")

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}


# =========================
# Pascal VOC bbox → YOLO bbox 변환
# =========================
def voc_to_yolo_bbox(size, box):
    img_w, img_h = size

    xmin, ymin, xmax, ymax = box

    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h

    return x_center, y_center, width, height


# =========================
# XML 파일을 YOLO txt로 변환
# =========================
def convert_xml_to_yolo(xml_path, label_output_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    yolo_lines = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip()

        if class_name not in CLASS_TO_ID:
            print(f"[경고] 알 수 없는 클래스: {class_name} / 파일: {xml_path}")
            continue

        class_id = CLASS_TO_ID[class_name]

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        x_center, y_center, width, height = voc_to_yolo_bbox(
            (img_w, img_h),
            (xmin, ymin, xmax, ymax),
        )

        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    label_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(label_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))


# =========================
# data.yaml 생성
# =========================
def create_data_yaml():
    yaml_path = OUTPUT_ROOT / "data.yaml"

    names_text = "\n".join([f"  {i}: {name}" for i, name in enumerate(CLASSES)])

    content = f"""path: {OUTPUT_ROOT.resolve()}
train: images/train
val: images/val

names:
{names_text}
"""

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[완료] data.yaml 생성: {yaml_path}")


# =========================
# 메인 처리
# =========================
def main():
    random.seed(RANDOM_SEED)

    all_items = []

    for class_folder in IMAGE_ROOT.iterdir():
        if not class_folder.is_dir():
            continue

        class_name_from_folder = class_folder.name

        for image_path in class_folder.glob("*.jpg"):
            xml_path = ANNOTATION_ROOT / class_name_from_folder / f"{image_path.stem}.xml"

            if not xml_path.exists():
                print(f"[경고] XML 없음: {xml_path}")
                continue

            all_items.append((image_path, xml_path))

    print(f"[정보] 전체 이미지 수: {len(all_items)}")

    random.shuffle(all_items)

    train_count = int(len(all_items) * TRAIN_RATIO)

    train_items = all_items[:train_count]
    val_items = all_items[train_count:]

    splits = {
        "train": train_items,
        "val": val_items,
    }

    for split_name, items in splits.items():
        image_output_dir = OUTPUT_ROOT / "images" / split_name
        label_output_dir = OUTPUT_ROOT / "labels" / split_name

        image_output_dir.mkdir(parents=True, exist_ok=True)
        label_output_dir.mkdir(parents=True, exist_ok=True)

        for image_path, xml_path in items:
            output_image_path = image_output_dir / image_path.name
            output_label_path = label_output_dir / f"{image_path.stem}.txt"

            shutil.copy2(image_path, output_image_path)
            convert_xml_to_yolo(xml_path, output_label_path)

        print(f"[완료] {split_name}: {len(items)}개 처리")

    create_data_yaml()

    print("\n전체 변환 완료")
    print(f"출력 폴더: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()