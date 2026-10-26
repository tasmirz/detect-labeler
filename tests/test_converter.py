import unittest
import json
import xml.etree.ElementTree as ET
from core.converter import AnnotationBox, UniversalConverter


class TestUniversalConverter(unittest.TestCase):
    def setUp(self):
        self.classes = ["car", "pedestrian", "traffic_light", "dog", "laptop"]
        self.img_w = 800
        self.img_h = 600

    def test_annotation_box_properties(self):
        box = AnnotationBox("car", 0, 100, 150, 300, 350, conf=0.95)
        self.assertEqual(box.width, 200)
        self.assertEqual(box.height, 200)
        d = box.to_dict()
        self.assertEqual(d["class_name"], "car")
        self.assertEqual(d["xmin"], 100.0)
        self.assertEqual(d["ymin"], 150.0)
        self.assertEqual(d["xmax"], 300.0)
        self.assertEqual(d["ymax"], 350.0)

    def test_yolo_roundtrip(self):
        original_boxes = [
            AnnotationBox("car", 0, 100, 150, 300, 350),
            AnnotationBox("pedestrian", 1, 400, 200, 450, 500)
        ]
        yolo_str = UniversalConverter.to_yolo(original_boxes, self.img_w, self.img_h)
        parsed_boxes = UniversalConverter.from_yolo(yolo_str, self.img_w, self.img_h, self.classes)

        self.assertEqual(len(parsed_boxes), 2)
        self.assertEqual(parsed_boxes[0].class_name, "car")
        self.assertAlmostEqual(parsed_boxes[0].xmin, 100.0, places=1)
        self.assertAlmostEqual(parsed_boxes[0].ymin, 150.0, places=1)
        self.assertAlmostEqual(parsed_boxes[0].xmax, 300.0, places=1)
        self.assertAlmostEqual(parsed_boxes[0].ymax, 350.0, places=1)

    def test_voc_roundtrip(self):
        original_boxes = [
            AnnotationBox("dog", 3, 50, 60, 250, 300),
            AnnotationBox("laptop", 4, 300, 100, 500, 400)
        ]
        voc_xml = UniversalConverter.to_voc(original_boxes, "test.jpg", self.img_w, self.img_h)
        parsed_boxes, w, h = UniversalConverter.from_voc(voc_xml)

        self.assertEqual(w, self.img_w)
        self.assertEqual(h, self.img_h)
        self.assertEqual(len(parsed_boxes), 2)
        self.assertEqual(parsed_boxes[0].class_name, "dog")
        self.assertEqual(parsed_boxes[0].xmin, 50.0)
        self.assertEqual(parsed_boxes[0].ymin, 60.0)
        self.assertEqual(parsed_boxes[0].xmax, 250.0)
        self.assertEqual(parsed_boxes[0].ymax, 300.0)

    def test_coco_roundtrip(self):
        images_data = {
            "sample.jpg": [
                AnnotationBox("car", 0, 10, 20, 110, 220),
                AnnotationBox("dog", 3, 200, 150, 350, 400)
            ]
        }
        image_dims = {"sample.jpg": (self.img_w, self.img_h)}
        coco_dict = UniversalConverter.to_coco(images_data, self.classes, image_dims)

        parsed_data = UniversalConverter.from_coco(coco_dict)
        self.assertIn("sample.jpg", parsed_data)
        boxes = parsed_data["sample.jpg"]
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0].class_name, "car")
        self.assertAlmostEqual(boxes[0].xmin, 10.0, places=1)
        self.assertAlmostEqual(boxes[0].ymin, 20.0, places=1)

    def test_csv_roundtrip(self):
        images_data = {
            "img1.jpg": [AnnotationBox("car", 0, 10, 20, 100, 200)],
            "img2.jpg": [AnnotationBox("dog", 3, 50, 60, 150, 250)]
        }
        csv_str = UniversalConverter.to_csv(images_data)
        parsed_data = UniversalConverter.from_csv(csv_str)

        self.assertIn("img1.jpg", parsed_data)
        self.assertIn("img2.jpg", parsed_data)
        self.assertEqual(len(parsed_data["img1.jpg"]), 1)
        self.assertEqual(parsed_data["img1.jpg"][0].class_name, "car")


if __name__ == "__main__":
    unittest.main()
