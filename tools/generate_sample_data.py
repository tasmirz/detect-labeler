#!/usr/bin/env python3
"""
Generate Multi-Domain Agnostic Sample Images and Annotations
============================================================
Creates 5 distinct domain images (Urban, Indoor, Wildlife, Retail, Aerial)
with accurate bounding box annotations in YOLO, COCO, and Pascal VOC formats.
"""

import os
import json
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "sample_data"
IMG_DIR = SAMPLE_DIR / "images"
ANN_DIR = SAMPLE_DIR / "annotations"
TAX_DIR = SAMPLE_DIR / "taxonomies"

for d in [IMG_DIR, ANN_DIR / "yolo", ANN_DIR / "coco", ANN_DIR / "voc", TAX_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def draw_styled_rect(img, x1, y1, x2, y2, color, fill_color, label):
    cv2.rectangle(img, (x1, y1), (x2, y2), fill_color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    # Inner detail
    cv2.rectangle(img, (x1 + 6, y1 + 6), (x2 - 6, y2 - 6), color, 1)
    # Label badge on image
    tw = len(label) * 9 + 10
    cv2.rectangle(img, (x1, max(0, y1 - 22)), (x1 + tw, y1), color, -1)
    cv2.putText(img, label, (x1 + 5, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def make_urban_scene():
    w, h = 960, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky to asphalt gradient
    for y in range(h):
        if y < 240:
            ratio = y / 240.0
            img[y, :] = [int(180 - ratio * 40), int(150 - ratio * 30), int(130 - ratio * 20)]
        elif y < 360:
            img[y, :] = [90, 85, 80]
        else:
            img[y, :] = [60, 60, 60]

    # Road lane markings
    for x in range(50, w, 120):
        cv2.rectangle(img, (x, 460), (x + 60, 475), (220, 220, 220), -1)

    # Objects to draw
    boxes = [
        {"class": "car", "cid": 0, "box": (120, 340, 360, 490), "col": (50, 120, 240), "fill": (30, 80, 180)},
        {"class": "car", "cid": 0, "box": (540, 310, 760, 440), "col": (60, 190, 60), "fill": (40, 130, 40)},
        {"class": "pedestrian", "cid": 1, "box": (420, 290, 490, 470), "col": (200, 70, 220), "fill": (140, 40, 160)},
        {"class": "traffic_light", "cid": 2, "box": (800, 110, 870, 270), "col": (230, 180, 30), "fill": (30, 30, 30)},
        {"class": "bicycle", "cid": 3, "box": (50, 380, 140, 480), "col": (40, 200, 220), "fill": (20, 140, 150)}
    ]
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        draw_styled_rect(img, x1, y1, x2, y2, b["col"], b["fill"], b["class"])
    return img, boxes, ["car", "pedestrian", "traffic_light", "bicycle"]


def make_indoor_scene():
    w, h = 960, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Wall and desk gradient
    img[:320, :] = [215, 210, 205]
    img[320:, :] = [130, 95, 65]

    boxes = [
        {"class": "monitor", "cid": 0, "box": (280, 140, 680, 410), "col": (220, 100, 50), "fill": (40, 40, 40)},
        {"class": "keyboard", "cid": 1, "box": (310, 450, 650, 540), "col": (50, 180, 120), "fill": (80, 80, 80)},
        {"class": "laptop", "cid": 2, "box": (80, 330, 290, 490), "col": (190, 80, 200), "fill": (120, 120, 130)},
        {"class": "coffee_mug", "cid": 3, "box": (720, 410, 810, 510), "col": (40, 140, 240), "fill": (220, 220, 230)},
        {"class": "desk_lamp", "cid": 4, "box": (750, 160, 890, 370), "col": (240, 200, 40), "fill": (180, 150, 30)}
    ]
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        draw_styled_rect(img, x1, y1, x2, y2, b["col"], b["fill"], b["class"])
    return img, boxes, ["monitor", "keyboard", "laptop", "coffee_mug", "desk_lamp"]


def make_wildlife_scene():
    w, h = 960, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Nature foliage backdrop
    img[:200, :] = [220, 180, 130]
    img[200:, :] = [60, 110, 45]

    boxes = [
        {"class": "deer", "cid": 0, "box": (310, 220, 580, 520), "col": (40, 120, 200), "fill": (80, 110, 140)},
        {"class": "bird", "cid": 1, "box": (160, 90, 270, 190), "col": (210, 160, 30), "fill": (160, 120, 20)},
        {"class": "tree_trunk", "cid": 2, "box": (710, 110, 890, 580), "col": (50, 80, 120), "fill": (40, 60, 90)},
        {"class": "boulder", "cid": 3, "box": (90, 440, 280, 570), "col": (140, 140, 140), "fill": (110, 110, 110)}
    ]
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        draw_styled_rect(img, x1, y1, x2, y2, b["col"], b["fill"], b["class"])
    return img, boxes, ["deer", "bird", "tree_trunk", "boulder"]


def make_retail_scene():
    w, h = 960, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Store shelf backdrop
    img[:, :] = [235, 235, 230]
    for sy in [220, 430]:
        cv2.rectangle(img, (0, sy), (w, sy + 18), (140, 140, 140), -1)

    boxes = [
        {"class": "cereal_box", "cid": 0, "box": (70, 60, 230, 215), "col": (50, 80, 230), "fill": (40, 60, 190)},
        {"class": "cereal_box", "cid": 0, "box": (250, 70, 400, 215), "col": (50, 80, 230), "fill": (30, 160, 180)},
        {"class": "soda_can", "cid": 1, "box": (440, 100, 530, 215), "col": (220, 50, 50), "fill": (180, 40, 40)},
        {"class": "juice_bottle", "cid": 2, "box": (560, 70, 680, 215), "col": (40, 180, 220), "fill": (30, 140, 170)},
        {"class": "snack_bag", "cid": 3, "box": (710, 270, 890, 425), "col": (200, 80, 180), "fill": (150, 50, 130)},
        {"class": "shampoo", "cid": 4, "box": (120, 260, 240, 425), "col": (60, 190, 80), "fill": (40, 150, 60)}
    ]
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        draw_styled_rect(img, x1, y1, x2, y2, b["col"], b["fill"], b["class"])
    return img, boxes, ["cereal_box", "soda_can", "juice_bottle", "snack_bag", "shampoo"]


def make_aerial_scene():
    w, h = 960, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Ocean harbor and pier surface
    img[:, :400] = [170, 120, 50]  # Deep blue-green ocean
    img[:, 400:] = [120, 125, 120] # Concrete harbor dock

    boxes = [
        {"class": "cargo_ship", "cid": 0, "box": (80, 120, 340, 490), "col": (200, 120, 50), "fill": (140, 80, 30)},
        {"class": "speedboat", "cid": 1, "box": (140, 520, 260, 610), "col": (220, 80, 210), "fill": (170, 50, 160)},
        {"class": "warehouse", "cid": 2, "box": (450, 90, 720, 310), "col": (80, 180, 60), "fill": (60, 140, 45)},
        {"class": "storage_tank", "cid": 3, "box": (770, 110, 910, 250), "col": (230, 170, 40), "fill": (180, 130, 20)},
        {"class": "semi_truck", "cid": 4, "box": (510, 410, 680, 530), "col": (40, 80, 220), "fill": (30, 60, 170)}
    ]
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        draw_styled_rect(img, x1, y1, x2, y2, b["col"], b["fill"], b["class"])
    return img, boxes, ["cargo_ship", "speedboat", "warehouse", "storage_tank", "semi_truck"]


def main():
    generators = [
        ("urban_street.jpg", make_urban_scene, "autonomous_driving.json"),
        ("indoor_office.jpg", make_indoor_scene, "indoor_electronics.json"),
        ("wildlife_nature.jpg", make_wildlife_scene, "wildlife_nature.json"),
        ("retail_products.jpg", make_retail_scene, "retail_inventory.json"),
        ("drone_aerial.jpg", make_aerial_scene, "aerial_harbor.json"),
    ]

    all_coco_images = []
    all_coco_annotations = []
    all_classes_set = set()
    ann_counter = 1

    for img_idx, (fname, gen_fn, tax_fname) in enumerate(generators, start=1):
        img_bgr, boxes, class_list = gen_fn()
        img_h, img_w = img_bgr.shape[:2]
        img_path = IMG_DIR / fname
        cv2.imwrite(str(img_path), img_bgr)
        print(f"Generated {fname} ({img_w}x{img_h}) with {len(boxes)} objects.")

        # Save domain taxonomy
        tax_path = TAX_DIR / tax_fname
        with open(tax_path, "w", encoding="utf-8") as f:
            json.dump({
                "taxonomy_name": tax_fname.replace(".json", "").replace("_", " ").title(),
                "classes": class_list
            }, f, indent=2)

        # 1. YOLO Format
        yolo_lines = []
        for b in boxes:
            x1, y1, x2, y2 = b["box"]
            xc = ((x1 + x2) / 2.0) / img_w
            yc = ((y1 + y2) / 2.0) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            cid = b["cid"]
            yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        stem = Path(fname).stem
        with open(ANN_DIR / "yolo" / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        # 2. Pascal VOC XML Format
        voc_lines = [
            "<annotation>",
            f"  <filename>{fname}</filename>",
            "  <size>",
            f"    <width>{img_w}</width>",
            f"    <height>{img_h}</height>",
            "    <depth>3</depth>",
            "  </size>"
        ]
        for b in boxes:
            x1, y1, x2, y2 = b["box"]
            voc_lines.extend([
                "  <object>",
                f"    <name>{b['class']}</name>",
                "    <pose>Unspecified</pose>",
                "    <truncated>0</truncated>",
                "    <difficult>0</difficult>",
                "    <bndbox>",
                f"      <xmin>{int(x1)}</xmin>",
                f"      <ymin>{int(y1)}</ymin>",
                f"      <xmax>{int(x2)}</xmax>",
                f"      <ymax>{int(y2)}</ymax>",
                "    </bndbox>",
                "  </object>"
            ])
        voc_lines.append("</annotation>")
        with open(ANN_DIR / "voc" / f"{stem}.xml", "w", encoding="utf-8") as f:
            f.write("\n".join(voc_lines))

        # 3. Collect for unified COCO
        all_coco_images.append({
            "id": img_idx,
            "file_name": fname,
            "width": img_w,
            "height": img_h
        })
        for b in boxes:
            all_classes_set.add(b["class"])
            x1, y1, x2, y2 = b["box"]
            all_coco_annotations.append({
                "id": ann_counter,
                "image_id": img_idx,
                "category_id": b["class"],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0
            })
            ann_counter += 1

    # Finalize COCO
    sorted_classes = sorted(list(all_classes_set))
    class_to_id = {c: i + 1 for i, c in enumerate(sorted_classes)}
    for a in all_coco_annotations:
        a["category_id"] = class_to_id[a["category_id"]]

    coco_payload = {
        "images": all_coco_images,
        "annotations": all_coco_annotations,
        "categories": [{"id": cid, "name": cname, "supercategory": "none"} for cname, cid in class_to_id.items()]
    }
    with open(ANN_DIR / "coco" / "sample_coco.json", "w", encoding="utf-8") as f:
        json.dump(coco_payload, f, indent=2)

    # Save generic COCO 80 taxonomy
    with open(TAX_DIR / "generic_coco80.json", "w", encoding="utf-8") as f:
        json.dump({
            "taxonomy_name": "COCO 80 Standard Objects",
            "classes": [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
            ]
        }, f, indent=2)

    print("✓ Successfully generated sample datasets and multi-domain taxonomies!")


if __name__ == "__main__":
    main()
