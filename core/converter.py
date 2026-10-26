"""
Core Format-Agnostic Annotation Converter
=========================================
Provides bidirectional conversion between standard computer vision formats:
- YOLO (normalized center-x, center-y, width, height)
- COCO JSON (images, annotations, categories with absolute [x, y, w, h])
- Pascal VOC XML (<annotation><object><name>...<bndbox><xmin>...<ymax>)
- CSV (filename, class_name, xmin, ymin, xmax, ymax)
"""

import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path


class AnnotationBox:
    def __init__(
        self,
        class_name: str,
        class_id: int,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        conf: float = 1.0,
        box_id: Optional[str] = None
    ):
        self.class_name = class_name
        self.class_id = class_id
        self.xmin = max(0.0, float(xmin))
        self.ymin = max(0.0, float(ymin))
        self.xmax = max(self.xmin + 1.0, float(xmax))
        self.ymax = max(self.ymin + 1.0, float(ymax))
        self.conf = float(conf)
        self.box_id = box_id

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    def to_yolo(self, img_w: int, img_h: int) -> str:
        """Converts box to YOLO normalized format: <class_id> <xc> <yc> <w> <h>."""
        xc = ((self.xmin + self.xmax) / 2.0) / img_w
        yc = ((self.ymin + self.ymax) / 2.0) / img_h
        w = self.width / img_w
        h = self.height / img_h
        return f"{self.class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "class_id": self.class_id,
            "xmin": round(self.xmin, 2),
            "ymin": round(self.ymin, 2),
            "xmax": round(self.xmax, 2),
            "ymax": round(self.ymax, 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "conf": round(self.conf, 4),
            "box_id": self.box_id
        }


class UniversalConverter:
    """Agnostic converter bridging YOLO, COCO, Pascal VOC, and CSV formats."""

    @staticmethod
    def from_yolo(
        content: str,
        img_w: int,
        img_h: int,
        classes: List[str]
    ) -> List[AnnotationBox]:
        """Parses YOLO normalized text into AnnotationBox objects."""
        boxes = []
        for line in content.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
                conf = float(parts[5]) if len(parts) >= 6 else 1.0

                abs_w = w * img_w
                abs_h = h * img_h
                xmin = (xc * img_w) - (abs_w / 2.0)
                ymin = (yc * img_h) - (abs_h / 2.0)
                xmax = xmin + abs_w
                ymax = ymin + abs_h

                cname = classes[cid] if 0 <= cid < len(classes) else f"class_{cid}"
                boxes.append(AnnotationBox(cname, cid, xmin, ymin, xmax, ymax, conf=conf))
            except Exception:
                continue
        return boxes

    @staticmethod
    def to_yolo(boxes: List[AnnotationBox], img_w: int, img_h: int) -> str:
        """Serializes list of AnnotationBox to YOLO format text."""
        lines = [b.to_yolo(img_w, img_h) for b in boxes]
        return "\n".join(lines)

    @staticmethod
    def from_voc(content: str) -> Tuple[List[AnnotationBox], int, int]:
        """Parses Pascal VOC XML into AnnotationBox list and image dimensions."""
        root = ET.fromstring(content)
        size_elem = root.find("size")
        img_w = int(size_elem.findtext("width", "640")) if size_elem is not None else 640
        img_h = int(size_elem.findtext("height", "640")) if size_elem is not None else 640

        boxes = []
        for obj in root.findall("object"):
            cname = obj.findtext("name", "object")
            bnd = obj.find("bndbox")
            if bnd is not None:
                xmin = float(bnd.findtext("xmin", "0"))
                ymin = float(bnd.findtext("ymin", "0"))
                xmax = float(bnd.findtext("xmax", "0"))
                ymax = float(bnd.findtext("ymax", "0"))
                boxes.append(AnnotationBox(cname, 0, xmin, ymin, xmax, ymax))
        return boxes, img_w, img_h

    @staticmethod
    def to_voc(
        boxes: List[AnnotationBox],
        filename: str,
        img_w: int,
        img_h: int,
        depth: int = 3
    ) -> str:
        """Serializes boxes into Pascal VOC XML."""
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = filename
        size_elem = ET.SubElement(root, "size")
        ET.SubElement(size_elem, "width").text = str(img_w)
        ET.SubElement(size_elem, "height").text = str(img_h)
        ET.SubElement(size_elem, "depth").text = str(depth)

        for b in boxes:
            obj_elem = ET.SubElement(root, "object")
            ET.SubElement(obj_elem, "name").text = b.class_name
            ET.SubElement(obj_elem, "pose").text = "Unspecified"
            ET.SubElement(obj_elem, "truncated").text = "0"
            ET.SubElement(obj_elem, "difficult").text = "0"

            bnd_elem = ET.SubElement(obj_elem, "bndbox")
            ET.SubElement(bnd_elem, "xmin").text = str(int(round(b.xmin)))
            ET.SubElement(bnd_elem, "ymin").text = str(int(round(b.ymin)))
            ET.SubElement(bnd_elem, "xmax").text = str(int(round(b.xmax)))
            ET.SubElement(bnd_elem, "ymax").text = str(int(round(b.ymax)))

        rough_string = ET.tostring(root, "utf-8")
        parsed = minidom.parseString(rough_string)
        return parsed.toprettyxml(indent="  ")

    @staticmethod
    def to_coco(
        images_and_boxes: Any,
        classes: List[str],
        image_dims: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> Dict[str, Any]:
        """
        Creates complete COCO dataset JSON from either:
        - List of dicts: [ { "file_name": str, "width": int, "height": int, "boxes": List[AnnotationBox] }, ... ]
        - Dict: { "file_name": List[AnnotationBox], ... } with image_dims
        """
        categories = [{"id": i, "name": name, "supercategory": "none"} for i, name in enumerate(classes)]
        images = []
        annotations = []
        ann_id = 1

        if isinstance(images_and_boxes, dict):
            items = []
            for fname, b_list in images_and_boxes.items():
                w, h = (image_dims.get(fname, (640, 640)) if image_dims else (640, 640))
                items.append({"file_name": fname, "width": w, "height": h, "boxes": b_list})
        else:
            items = images_and_boxes

        for img_idx, item in enumerate(items, 1):
            img_id = item.get("image_id", img_idx)
            images.append({
                "id": img_id,
                "file_name": item["file_name"],
                "width": item["width"],
                "height": item["height"]
            })

            for b in item.get("boxes", []):
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": b.class_id,
                    "bbox": [round(b.xmin, 2), round(b.ymin, 2), round(b.width, 2), round(b.height, 2)],
                    "area": round(b.width * b.height, 2),
                    "segmentation": [],
                    "iscrowd": 0,
                    "score": b.conf
                })
                ann_id += 1

        return {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }

    @staticmethod
    def from_coco(coco_dict: Dict[str, Any]) -> Dict[str, List[AnnotationBox]]:
        """Parses COCO JSON dict into mapping of filename -> List[AnnotationBox]."""
        cat_map = {c["id"]: c["name"] for c in coco_dict.get("categories", [])}
        img_map = {img["id"]: img["file_name"] for img in coco_dict.get("images", [])}

        result: Dict[str, List[AnnotationBox]] = {fname: [] for fname in img_map.values()}
        for ann in coco_dict.get("annotations", []):
            img_id = ann.get("image_id")
            fname = img_map.get(img_id)
            if not fname:
                continue

            cid = ann.get("category_id", 0)
            cname = cat_map.get(cid, f"class_{cid}")
            bbox = ann.get("bbox", [0, 0, 0, 0])
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h
            conf = float(ann.get("score", 1.0))
            box = AnnotationBox(cname, cid, xmin, ymin, xmax, ymax, conf=conf)
            result[fname].append(box)

        return result

    @staticmethod
    def to_csv(images_and_boxes: Any) -> str:
        """Exports annotations to standard CSV format."""
        lines = ["filename,class_id,class_name,xmin,ymin,xmax,ymax,confidence"]
        if isinstance(images_and_boxes, dict):
            for fname, b_list in images_and_boxes.items():
                for b in b_list:
                    lines.append(f"{fname},{b.class_id},{b.class_name},{b.xmin:.2f},{b.ymin:.2f},{b.xmax:.2f},{b.ymax:.2f},{b.conf:.4f}")
        else:
            for item in images_and_boxes:
                fname = item["file_name"]
                for b in item.get("boxes", []):
                    lines.append(f"{fname},{b.class_id},{b.class_name},{b.xmin:.2f},{b.ymin:.2f},{b.xmax:.2f},{b.ymax:.2f},{b.conf:.4f}")
        return "\n".join(lines)

    @staticmethod
    def from_csv(csv_content: str) -> Dict[str, List[AnnotationBox]]:
        """Parses CSV format into mapping of filename -> List[AnnotationBox]."""
        result: Dict[str, List[AnnotationBox]] = {}
        lines = csv_content.strip().splitlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line or (idx == 0 and "filename" in line.lower()):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            fname = parts[0]
            try:
                cid = int(parts[1])
                cname = parts[2]
                xmin = float(parts[3])
                ymin = float(parts[4])
                xmax = float(parts[5])
                ymax = float(parts[6])
                conf = float(parts[7]) if len(parts) >= 8 else 1.0
                box = AnnotationBox(cname, cid, xmin, ymin, xmax, ymax, conf=conf)
                if fname not in result:
                    result[fname] = []
                result[fname].append(box)
            except Exception:
                continue
        return result

