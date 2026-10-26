import unittest
import numpy as np
import cv2
from core.smart_annotator import SmartAnnotator


class TestSmartAnnotator(unittest.TestCase):
    def setUp(self):
        # Create a synthetic image with a solid rectangle
        self.img = np.zeros((400, 400, 3), dtype=np.uint8)
        # Draw a white rectangle at (100, 100) to (200, 200)
        cv2.rectangle(self.img, (100, 100), (200, 200), (255, 255, 255), -1)

    def test_snap_box_to_contour(self):
        # Provide a loose box around the rectangle
        rough_box = (90, 90, 210, 210)
        snapped = SmartAnnotator.snap_box_to_contour(self.img, rough_box)

        # The snapped box should align closely with (100, 100, 200, 200)
        self.assertAlmostEqual(snapped[0], 100, delta=5)
        self.assertAlmostEqual(snapped[1], 100, delta=5)
        self.assertAlmostEqual(snapped[2], 200, delta=5)
        self.assertAlmostEqual(snapped[3], 200, delta=5)

    def test_click_to_box(self):
        # Click near the center of the rectangle: (150, 150)
        box = SmartAnnotator.click_to_box(self.img, 150, 150)
        self.assertIsNotNone(box)
        x1, y1, x2, y2 = box
        self.assertAlmostEqual(x1, 100, delta=5)
        self.assertAlmostEqual(y1, 100, delta=5)
        self.assertAlmostEqual(x2, 200, delta=5)
        self.assertAlmostEqual(y2, 200, delta=5)

    def test_salient_proposals(self):
        proposals = SmartAnnotator.extract_salient_proposals(self.img, max_proposals=5)
        self.assertGreaterEqual(len(proposals), 1)
        # One of the proposals should contain our rectangle
        x1, y1, x2, y2 = proposals[0]
        self.assertAlmostEqual(x1, 100, delta=10)
        self.assertAlmostEqual(y1, 100, delta=10)


if __name__ == "__main__":
    unittest.main()
