"""
detect-labeler: Universal Domain-Agnostic Image Labeling Application
===================================================================
A fully decoupled, format-agnostic, and model-agnostic computer vision annotation platform.
"""

import os
import io
import json
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from core.converter import AnnotationBox, UniversalConverter
from core.smart_annotator import SmartAnnotator
from core.model_adapter import ModelAdapter

app = FastAPI(
    title="detect-labeler",
    description="Universal Domain-Agnostic Computer Vision Annotation Platform",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
SAMPLE_DIR = BASE_DIR / "sample_data"
IMAGES_DIR = SAMPLE_DIR / "images"
TAXONOMIES_DIR = SAMPLE_DIR / "taxonomies"
ANNOTATIONS_DIR = SAMPLE_DIR / "annotations"

# Ensure directories exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TAXONOMIES_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# In-memory storage for active session annotations
SESSION_ANNOTATIONS: Dict[str, List[AnnotationBox]] = {}
ACTIVE_TAXONOMY: List[str] = [
    "object", "person", "vehicle", "animal", "equipment"
]


# Models
class BoxPayload(BaseModel):
    class_name: str
    class_id: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    conf: float = 1.0


class AnnotationsSaveRequest(BaseModel):
    format: str  # yolo, voc, coco, csv
    boxes: List[BoxPayload]
    img_w: int
    img_h: int


class SnapRequest(BaseModel):
    image_name: str
    rough_box: List[float]  # [xmin, ymin, xmax, ymax]
    padding: float = 0.20


class ClickRequest(BaseModel):
    image_name: str
    click_x: int
    click_y: int
    radius: int = 150


class TaxonomyPayload(BaseModel):
    name: str
    classes: List[str]


def load_image_bgr(filename: str) -> Optional[np.ndarray]:
    img_path = IMAGES_DIR / filename
    if not img_path.exists():
        return None
    return cv2.imread(str(img_path))


@app.on_event("startup")
def startup_event():
    # Pre-populate session with any sample annotations present
    yolo_dir = ANNOTATIONS_DIR / "yolo"
    if yolo_dir.exists() and IMAGES_DIR.exists():
        for img_path in IMAGES_DIR.glob("*.jpg"):
            txt_path = yolo_dir / f"{img_path.stem}.txt"
            if txt_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    boxes = UniversalConverter.from_yolo(
                        txt_path.read_text(encoding="utf-8"),
                        w, h,
                        ACTIVE_TAXONOMY
                    )
                    SESSION_ANNOTATIONS[img_path.name] = boxes


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/images")
def list_images():
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for p in IMAGES_DIR.glob(ext):
            ann_count = len(SESSION_ANNOTATIONS.get(p.name, []))
            images.append({
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "has_annotations": ann_count > 0,
                "box_count": ann_count
            })
    images.sort(key=lambda x: x["filename"])
    return {"images": images}


@app.get("/api/image/{filename}")
def get_image(filename: str):
    path = IMAGES_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...)):
    target_path = IMAGES_DIR / file.filename
    content = await file.read()
    target_path.write_bytes(content)
    return {"status": "success", "filename": file.filename}


@app.get("/api/taxonomies")
def list_taxonomies():
    taxonomies = []
    for p in TAXONOMIES_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            taxonomies.append({
                "id": p.stem,
                "name": data.get("domain", p.stem).replace("_", " ").title(),
                "classes": data.get("classes", [])
            })
        except Exception:
            continue
    return {
        "active": ACTIVE_TAXONOMY,
        "presets": taxonomies
    }


@app.post("/api/taxonomies/set_active")
def set_active_taxonomy(payload: TaxonomyPayload):
    global ACTIVE_TAXONOMY
    if not payload.classes:
        raise HTTPException(status_code=400, detail="Classes list cannot be empty")
    ACTIVE_TAXONOMY = payload.classes
    return {"status": "success", "active": ACTIVE_TAXONOMY}


@app.post("/api/taxonomies/save_preset")
def save_preset_taxonomy(payload: TaxonomyPayload):
    safe_name = payload.name.lower().replace(" ", "_")
    target = TAXONOMIES_DIR / f"{safe_name}.json"
    data = {
        "domain": safe_name,
        "classes": payload.classes
    }
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"status": "success", "id": safe_name}


@app.get("/api/annotations/{filename}")
def get_annotations(filename: str):
    boxes = SESSION_ANNOTATIONS.get(filename, [])
    # Get image dimensions
    img_path = IMAGES_DIR / filename
    w, h = 640, 640
    if img_path.exists():
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]

    return {
        "filename": filename,
        "width": w,
        "height": h,
        "boxes": [b.to_dict() for b in boxes]
    }


@app.post("/api/annotations/{filename}")
def save_annotations(filename: str, payload: AnnotationsSaveRequest):
    boxes = [
        AnnotationBox(
            class_name=b.class_name,
            class_id=b.class_id,
            xmin=b.xmin,
            ymin=b.ymin,
            xmax=b.xmax,
            ymax=b.ymax,
            conf=b.conf
        )
        for b in payload.boxes
    ]
    SESSION_ANNOTATIONS[filename] = boxes

    # Persist locally based on requested format
    fmt = payload.format.lower()
    stem = Path(filename).stem

    if fmt == "yolo":
        yolo_dir = ANNOTATIONS_DIR / "yolo"
        yolo_dir.mkdir(parents=True, exist_ok=True)
        txt = UniversalConverter.to_yolo(boxes, payload.img_w, payload.img_h)
        (yolo_dir / f"{stem}.txt").write_text(txt, encoding="utf-8")

    elif fmt == "voc":
        voc_dir = ANNOTATIONS_DIR / "voc"
        voc_dir.mkdir(parents=True, exist_ok=True)
        xml = UniversalConverter.to_voc(boxes, filename, payload.img_w, payload.img_h)
        (voc_dir / f"{stem}.xml").write_text(xml, encoding="utf-8")

    return {"status": "saved", "count": len(boxes), "format": fmt}


@app.post("/api/snap_box")
def snap_box(payload: SnapRequest):
    img = load_image_bgr(payload.image_name)
    if img is None:
        raise HTTPException(status_code=404, detail="Image not found")

    snapped = SmartAnnotator.snap_box_to_contour(
        img,
        tuple(payload.rough_box),
        padding_ratio=payload.padding
    )
    return {"snapped_box": list(snapped)}


@app.post("/api/click_box")
def click_box(payload: ClickRequest):
    img = load_image_bgr(payload.image_name)
    if img is None:
        raise HTTPException(status_code=404, detail="Image not found")

    box = SmartAnnotator.click_to_box(
        img,
        payload.click_x,
        payload.click_y,
        search_radius=payload.radius
    )
    return {"box": list(box)}


@app.post("/api/salient_proposals")
def salient_proposals(payload: Dict[str, Any]):
    filename = payload.get("image_name")
    img = load_image_bgr(filename)
    if img is None:
        raise HTTPException(status_code=404, detail="Image not found")

    proposals = SmartAnnotator.extract_salient_proposals(
        img,
        max_proposals=payload.get("max_proposals", 15)
    )
    return {"proposals": [list(p) for p in proposals]}


@app.get("/api/export_dataset")
def export_dataset(format: str = "yolo"):
    """Bundles entire labeled dataset into a portable ZIP package."""
    zip_buffer = io.BytesIO()
    fmt = format.lower()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Images
        for img_path in IMAGES_DIR.glob("*.*"):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                zf.write(img_path, f"images/{img_path.name}")

        # 2. Annotations based on format
        if fmt == "yolo":
            for fname, boxes in SESSION_ANNOTATIONS.items():
                img_path = IMAGES_DIR / fname
                w, h = 640, 640
                if img_path.exists():
                    im = cv2.imread(str(img_path))
                    if im is not None:
                        h, w = im.shape[:2]
                yolo_txt = UniversalConverter.to_yolo(boxes, w, h)
                zf.writestr(f"labels/{Path(fname).stem}.txt", yolo_txt)

            # Add data.yaml and classes.txt
            yaml_content = f"names:\n" + "\n".join([f"  {i}: {name}" for i, name in enumerate(ACTIVE_TAXONOMY)])
            zf.writestr("data.yaml", yaml_content)
            zf.writestr("classes.txt", "\n".join(ACTIVE_TAXONOMY))

        elif fmt == "coco":
            img_items = []
            for fname, boxes in SESSION_ANNOTATIONS.items():
                img_path = IMAGES_DIR / fname
                w, h = 640, 640
                if img_path.exists():
                    im = cv2.imread(str(img_path))
                    if im is not None:
                        h, w = im.shape[:2]
                img_items.append({"file_name": fname, "width": w, "height": h, "boxes": boxes})

            coco_dict = UniversalConverter.to_coco(img_items, ACTIVE_TAXONOMY)
            zf.writestr("annotations/instances.json", json.dumps(coco_dict, indent=2))

        elif fmt == "voc":
            for fname, boxes in SESSION_ANNOTATIONS.items():
                img_path = IMAGES_DIR / fname
                w, h = 640, 640
                if img_path.exists():
                    im = cv2.imread(str(img_path))
                    if im is not None:
                        h, w = im.shape[:2]
                voc_xml = UniversalConverter.to_voc(boxes, fname, w, h)
                zf.writestr(f"annotations/{Path(fname).stem}.xml", voc_xml)

        elif fmt == "csv":
            img_items = []
            for fname, boxes in SESSION_ANNOTATIONS.items():
                img_items.append({"file_name": fname, "boxes": boxes})
            csv_str = UniversalConverter.to_csv(img_items)
            zf.writestr("annotations.csv", csv_str)

        # Include README explaining how to use the exported data
        readme = f"""# detect-labeler Exported Dataset
Format: {fmt.upper()}
Taxonomy: {len(ACTIVE_TAXONOMY)} classes ({', '.join(ACTIVE_TAXONOMY)})
Labeled Images: {len(SESSION_ANNOTATIONS)}
Generated by detect-labeler (Universal Domain-Agnostic Computer Vision Annotation Platform)
"""
        zf.writestr("DATASET_INFO.md", readme)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=detect_labeler_{fmt}_export.zip"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
