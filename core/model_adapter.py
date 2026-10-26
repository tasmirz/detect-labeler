"""
Agnostic Model Inference Adapter
================================
Provides a plug-and-play hook for machine learning models (YOLO / ONNX / PyTorch)
to generate assisted candidate detections across arbitrary classes and domains.
Gracefully handles environments where deep learning libraries are not installed.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np


class ModelAdapter:
    """Universal detector interface allowing users to hook any model for assisted pre-labeling."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.backend_type = "none"
        self.classes: List[str] = []

        if model_path and Path(model_path).is_file():
            self._load_model(model_path)

    def _load_model(self, path: str):
        path_obj = Path(path)
        if path_obj.suffix.lower() == ".pt":
            try:
                from ultralytics import YOLO
                self.model = YOLO(path)
                self.backend_type = "yolo"
                self.classes = list(self.model.names.values()) if hasattr(self.model, "names") else []
                print(f"[ModelAdapter] Loaded YOLO model from {path} with {len(self.classes)} classes.")
            except ImportError:
                print("[ModelAdapter] 'ultralytics' library not installed. Falling back to CV annotator.")
            except Exception as e:
                print(f"[ModelAdapter] Failed loading YOLO model: {e}")

        elif path_obj.suffix.lower() == ".onnx":
            try:
                import cv2
                self.model = cv2.dnn.readNetFromONNX(path)
                self.backend_type = "onnx_dnn"
                print(f"[ModelAdapter] Loaded ONNX model via OpenCV DNN from {path}.")
            except Exception as e:
                print(f"[ModelAdapter] Failed loading ONNX model: {e}")

    def predict(
        self,
        img_bgr: np.ndarray,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Runs model inference if loaded, returning a list of detections:
        [{"xmin": float, "ymin": float, "xmax": float, "ymax": float, "class_id": int, "class_name": str, "conf": float}]
        """
        if self.model is None or self.backend_type == "none":
            return []

        h, w = img_bgr.shape[:2]
        detections = []

        if self.backend_type == "yolo":
            try:
                results = self.model.predict(img_bgr, conf=conf_thresh, iou=iou_thresh, verbose=False)
                if results and len(results) > 0:
                    r = results[0]
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    cids = r.boxes.cls.cpu().numpy().astype(int)
                    names = r.names if hasattr(r, "names") else {}

                    for box, c, cid in zip(boxes, confs, cids):
                        x1, y1, x2, y2 = box
                        cname = names.get(cid, f"class_{cid}")
                        detections.append({
                            "xmin": float(x1),
                            "ymin": float(y1),
                            "xmax": float(x2),
                            "ymax": float(y2),
                            "class_id": int(cid),
                            "class_name": str(cname),
                            "conf": float(c)
                        })
            except Exception as e:
                print(f"[ModelAdapter] Inference error: {e}")

        return detections

    def is_available(self) -> bool:
        return self.model is not None
