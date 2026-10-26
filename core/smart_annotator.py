"""
Core Smart Annotator & Object Snapping Engine
=============================================
Provides model-agnostic computer vision mechanisms:
1. Snap-to-Contour: Refines a loose user box to exact object boundaries using Otsu & Canny edge detection.
2. Click-to-Box: Automatically generates a bounding box around an object when the user clicks its center.
3. Salient Proposal Extractor: Generates candidate object bounding boxes across the frame without needing a neural network.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class SmartAnnotator:
    """Agnostic OpenCV-based assisted annotation engine."""

    @staticmethod
    def snap_box_to_contour(
        img_bgr: np.ndarray,
        rough_box: Tuple[float, float, float, float],
        padding_ratio: float = 0.20
    ) -> Tuple[float, float, float, float]:
        """
        Refines a loose user bounding box (xmin, ymin, xmax, ymax) by analyzing
        internal edge gradients and contour contours within the padded region.
        """
        h_img, w_img = img_bgr.shape[:2]
        rx1, ry1, rx2, ry2 = rough_box
        bw = rx2 - rx1
        bh = ry2 - ry1

        if bw < 5 or bh < 5:
            return rough_box

        # Expand region slightly for context
        px = max(0, int(rx1 - bw * padding_ratio))
        py = max(0, int(ry1 - bh * padding_ratio))
        px2 = min(w_img, int(rx2 + bw * padding_ratio))
        py2 = min(h_img, int(ry2 + bh * padding_ratio))

        roi = img_bgr[py:py2, px:px2]
        if roi.size == 0:
            return rough_box

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Multi-strategy contour detection: Otsu threshold with polarity check + Canny edges
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        border_mean = (otsu[0, :].mean() + otsu[-1, :].mean() + otsu[:, 0].mean() + otsu[:, -1].mean()) / 4.0
        if border_mean > 127:
            otsu = cv2.bitwise_not(otsu)

        edges = cv2.Canny(blurred, 50, 150)
        combined = cv2.bitwise_or(otsu, edges)

        # Morphological closing to bridge fragmented contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return rough_box

        # Target center in local ROI coordinates
        target_cx = (rx1 + rx2) / 2.0 - px
        target_cy = (ry1 + ry2) / 2.0 - py

        best_box = None
        best_score = -1.0

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < 6 or h < 6:
                continue

            # Candidate center
            cx = x + w / 2.0
            cy = y + h / 2.0
            dist = np.hypot(cx - target_cx, cy - target_cy)
            area = w * h

            # Prioritize contours close to target center and covering a good fraction of rough box
            rough_area = bw * bh
            coverage = min(area, rough_area) / max(area, rough_area)
            score = (1.0 / (1.0 + dist * 0.05)) * (0.5 + 0.5 * coverage)

            if score > best_score:
                best_score = score
                best_box = (px + x, py + y, px + x + w, py + y + h)

        return best_box if best_box is not None else rough_box

    @staticmethod
    def click_to_box(
        img_bgr: np.ndarray,
        click_x: int,
        click_y: int,
        search_radius: int = 150
    ) -> Tuple[float, float, float, float]:
        """
        Generates an object bounding box around a single click location using FloodFill
        and gradient boundary analysis.
        """
        h_img, w_img = img_bgr.shape[:2]
        x1 = max(0, click_x - search_radius)
        y1 = max(0, click_y - search_radius)
        x2 = min(w_img, click_x + search_radius)
        y2 = min(h_img, click_y + search_radius)

        roi = img_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return (max(0, click_x - 30), max(0, click_y - 30), min(w_img, click_x + 30), min(h_img, click_y + 30))

        local_cx = click_x - x1
        local_cy = click_y - y1
        h_roi, w_roi = roi.shape[:2]

        # Method 1: FloodFill segmentation from click point
        try:
            ff_mask = np.zeros((h_roi + 2, w_roi + 2), dtype=np.uint8)
            diff = 15
            cv2.floodFill(
                roi.copy(),
                ff_mask,
                (local_cx, local_cy),
                (0, 255, 0),
                (diff, diff, diff),
                (diff, diff, diff),
                cv2.FLOODFILL_MASK_ONLY | (255 << 8)
            )
            ff_region = ff_mask[1:-1, 1:-1]
            cnts, _ = cv2.findContours(ff_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(c)
                if bw >= 8 and bh >= 8:
                    return (float(x1 + bx), float(y1 + by), float(x1 + bx + bw), float(y1 + by + bh))
        except Exception:
            pass

        # Method 2: Adaptive thresholding & edge boundaries
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bx <= local_cx <= bx + bw and by <= local_cy <= by + bh:
                candidates.append((bx, by, bw, bh))

        if candidates:
            bx, by, bw, bh = max(candidates, key=lambda b: b[2] * b[3])
            return (float(x1 + bx), float(y1 + by), float(x1 + bx + bw), float(y1 + by + bh))

        # Fallback to snap around default seed
        seed_box = (max(0, click_x - 40), max(0, click_y - 40), min(w_img, click_x + 40), min(h_img, click_y + 40))
        return SmartAnnotator.snap_box_to_contour(img_bgr, seed_box)

    @staticmethod
    def extract_salient_proposals(
        img_bgr: np.ndarray,
        max_proposals: int = 25,
        min_area: int = 400
    ) -> List[Tuple[float, float, float, float]]:
        """
        Extracts salient candidate bounding boxes across the full frame using
        multi-scale gradient energy and contour bounding. Useful for cold-start labeling.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        norm_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        _, thresh = cv2.threshold(norm_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        img_area = w * h

        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if min_area <= area <= img_area * 0.85:
                aspect = bw / float(bh)
                if 0.1 <= aspect <= 10.0:
                    boxes.append((float(x), float(y), float(x + bw), float(y + bh), area))

        # Sort by area descending and apply Non-Maximum Suppression
        boxes.sort(key=lambda b: b[4], reverse=True)
        final_boxes = []

        for b in boxes:
            x1, y1, x2, y2, _ = b
            keep = True
            for fx1, fy1, fx2, fy2 in final_boxes:
                # Calculate IoU
                inter_x1 = max(x1, fx1)
                inter_y1 = max(y1, fy1)
                inter_x2 = min(x2, fx2)
                inter_y2 = min(y2, fy2)
                inter_w = max(0.0, inter_x2 - inter_x1)
                inter_h = max(0.0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                union_area = ((x2 - x1) * (y2 - y1)) + ((fx2 - fx1) * (fy2 - fy1)) - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                if iou > 0.40:
                    keep = False
                    break
            if keep:
                final_boxes.append((x1, y1, x2, y2))
                if len(final_boxes) >= max_proposals:
                    break

        return final_boxes
