import unittest
from fastapi.testclient import TestClient
from app import app


class TestDetectLabelerAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_get_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("detect-labeler", response.text)

    def test_get_images(self):
        response = self.client.get("/api/images")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("images", data)
        self.assertGreaterEqual(len(data["images"]), 1)

    def test_get_taxonomies(self):
        response = self.client.get("/api/taxonomies")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("active", data)
        self.assertIn("presets", data)
        self.assertGreaterEqual(len(data["presets"]), 1)

    def test_snap_box_api(self):
        # urban_street.jpg was generated in sample_data
        response = self.client.post("/api/snap_box", json={
            "image_name": "urban_street.jpg",
            "rough_box": [100.0, 100.0, 300.0, 300.0],
            "padding": 0.20
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("snapped_box", data)
        self.assertEqual(len(data["snapped_box"]), 4)

    def test_click_box_api(self):
        response = self.client.post("/api/click_box", json={
            "image_name": "urban_street.jpg",
            "click_x": 200,
            "click_y": 200,
            "radius": 150
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("box", data)
        self.assertEqual(len(data["box"]), 4)

    def test_salient_proposals_api(self):
        response = self.client.post("/api/salient_proposals", json={
            "image_name": "urban_street.jpg",
            "max_proposals": 5
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("proposals", data)
        self.assertGreaterEqual(len(data["proposals"]), 1)

    def test_export_dataset_zip(self):
        response = self.client.get("/api/export_dataset?format=yolo")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/zip")


if __name__ == "__main__":
    unittest.main()
