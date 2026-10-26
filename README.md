# detect-labeler

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-5C3EE8.svg)](https://opencv.org/)

> **Universal, domain-agnostic computer vision annotation platform** with smart contour snapping, click-to-box segmentation, and multi-format conversion (YOLO, COCO, Pascal VOC, CSV).

---

## 🌟 Why detect-labeler?

Most image labeling tools are tightly coupled to specific domains, hardcoded class registries, or rigid directory hierarchies. **detect-labeler** was architected from the ground up to be **100% domain-agnostic, format-agnostic, and model-agnostic**:

- **No Hardcoded Taxonomy**: Freely create, rename, re-color, import (`classes.txt`, `data.yaml`, `.json`), or export label sets for any computer vision task.
- **Universal Format Interoperability**: Seamless bidirectional conversion between **YOLO**, **COCO JSON**, **Pascal VOC XML**, and **CSV**.
- **Model-Free Smart Assistance**: Snaps loose boxes to exact object contours and segments objects from a single click using high-speed OpenCV computer vision (Otsu adaptive thresholding + Canny edge gradients + FloodFill) without needing a GPU or heavyweight deep learning dependencies.
- **Plug-and-Play Model Hook**: Ready-to-use adapter to connect any trained YOLO (`.pt`), ONNX, or TorchScript model for assisted pre-labeling.

---

## 🛠️ Concrete Annotation Mechanisms

| Mechanism | Description | Trigger |
| :--- | :--- | :---: |
| **Manual Bounding Box** | Drag-to-draw boxes with 8 interactive resize handles, box repositioning, and pan/zoom canvas. | <kbd>B</kbd> |
| **Smart Contour Snapping** | Draw a loose rough box around an object; the engine detects edge gradients and Otsu polarity to snap tightly to boundaries. | <kbd>S</kbd> |
| **Click-to-Box** | Single click on any object; local FloodFill and boundary tracing automatically produce the exact bounding box. | <kbd>C</kbd> |
| **Salient Auto-Proposals** | Scans the frame for high-energy spatial gradients and extracts candidate object proposals cold-start. | <kbd>A</kbd> |
| **Model-Assisted Inference** | Optional hook for batch or real-time inference via YOLO (`.pt`) or ONNX models. | API Hook |

---

## 📸 Multi-Domain Sample Datasets

Out of the box, `detect-labeler` ships with synthesized real-world samples and multi-domain taxonomies across 5 distinct domains:

1. **Urban Street (Autonomous Driving)**: Vehicles, pedestrians, cyclists, traffic signals, road infrastructure.
2. **Indoor Smart Office (Electronics)**: Laptops, monitors, smartphones, keyboards, desk accessories.
3. **Wildlife & Nature Conservation**: Birds, mammals, natural flora, ecological tracking.
4. **Retail Inventory & Monitoring**: Packaged goods, beverage bottles, cereal boxes, shelf items.
5. **Aerial Drone & Maritime Harbor**: Ships, containers, cranes, docks, maritime infrastructure.

All sample photos and ground-truth labels in YOLO, COCO, and Pascal VOC formats are located in `sample_data/`.

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/tasmirz/detect-labeler.git
cd detect-labeler

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Sample Datasets & Taxonomies

```bash
python tools/generate_sample_data.py
```

### 3. Launch the Annotation Studio

```bash
python app.py
```
Open your browser at **`http://localhost:8000`**.

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
| :---: | :--- |
| <kbd>1</kbd> - <kbd>9</kbd> | Quick switch active class from taxonomy palette |
| <kbd>B</kbd> | Activate Manual Bounding Box tool |
| <kbd>S</kbd> | Activate Smart Contour Snapping tool |
| <kbd>C</kbd> | Activate Click-to-Box tool |
| <kbd>A</kbd> | Run Salient Auto-Proposal generator |
| <kbd>Del</kbd> / <kbd>Backspace</kbd> | Delete currently selected bounding box |
| <kbd>Ctrl</kbd> + <kbd>Z</kbd> | Undo last annotation edit |
| <kbd>Ctrl</kbd> + <kbd>S</kbd> | Save current frame annotations to disk |
| <kbd>Space</kbd> + Drag | Pan across zoomed canvas |
| <kbd>Mouse Wheel</kbd> | Smooth zoom centered on cursor |

---

## 🔄 Universal Format Conversion API

Convert annotations programmatically in Python:

```python
from core.converter import UniversalConverter, AnnotationBox

# 1. Parse YOLO format
boxes = UniversalConverter.from_yolo(
    content="0 0.50 0.50 0.20 0.30",
    img_w=1920,
    img_h=1080,
    classes=["car", "pedestrian"]
)

# 2. Export to Pascal VOC XML
voc_xml = UniversalConverter.to_voc(boxes, "frame_001.jpg", 1920, 1080)

# 3. Export to COCO JSON
coco_json = UniversalConverter.to_coco(
    images_and_boxes=[{"file_name": "frame_001.jpg", "width": 1920, "height": 1080, "boxes": boxes}],
    classes=["car", "pedestrian"]
)

# 4. Export to CSV
csv_text = UniversalConverter.to_csv([{"file_name": "frame_001.jpg", "boxes": boxes}])
```

---

## 🧪 Running the Test Suite

```bash
python -m unittest discover tests
```

---

## 📁 Repository Structure

```
detect-labeler/
├── app.py                      # FastAPI server & REST API
├── core/
│   ├── __init__.py             # Core package initialization
│   ├── converter.py            # Universal format converter (YOLO, COCO, VOC, CSV)
│   ├── smart_annotator.py      # OpenCV contour snapping & click-to-box engine
│   └── model_adapter.py        # Agnostic model inference adapter
├── static/
│   ├── css/style.css           # Modern dark-theme responsive UI styling
│   └── js/
│       ├── canvas.js           # HTML5 Canvas 60fps rendering & interaction engine
│       └── app.js              # State manager, hotkeys, and REST API controller
├── templates/
│   └── index.html              # Annotation studio web workspace
├── tools/
│   └── generate_sample_data.py # Multi-domain synthetic generator
├── sample_data/
│   ├── images/                 # Sample images across 5 distinct domains
│   ├── annotations/            # Ground truth in YOLO, COCO, and Pascal VOC
│   └── taxonomies/             # Pre-configured class lists for multiple domains
├── tests/
│   ├── test_converter.py       # Unit tests for format conversion
│   ├── test_smart_annotator.py # Unit tests for CV snapping mechanisms
│   └── test_api.py             # Integration tests for FastAPI endpoints
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # Documentation
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
