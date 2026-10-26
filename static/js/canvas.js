/**
 * detect-labeler: High-Precision HTML5 Canvas Annotation Engine
 * =============================================================
 * Handles rendering, pan/zoom, interactive box creation, 8-handle resizing,
 * crosshairs, and integration with smart contour snapping & click-to-box.
 */

class AnnotationCanvas {
  constructor(canvasElement, viewportElement) {
    this.canvas = canvasElement;
    this.viewport = viewportElement;
    this.ctx = this.canvas.getContext('2d');

    // Image & Scaling State
    this.image = null;
    this.imageWidth = 0;
    this.imageHeight = 0;
    this.scale = 1.0;
    this.offsetX = 0;
    this.offsetY = 0;

    // Annotation Data
    this.boxes = [];
    this.selectedBoxIndex = -1;
    this.history = []; // Undo stack

    // Interaction Modes: 'manual', 'snap', 'click', 'pan'
    this.currentTool = 'manual';
    this.showCrosshairs = true;

    // Mouse Interaction State
    this.isMouseDown = false;
    this.isPanning = false;
    this.isDrawing = false;
    this.isDraggingBox = false;
    this.isResizing = false;
    this.activeHandle = null; // 'tl', 'tr', 'bl', 'br', 'tc', 'bc', 'lc', 'rc'

    this.dragStart = { x: 0, y: 0 };
    this.currentMouse = { imgX: 0, imgY: 0, screenX: 0, screenY: 0 };
    this.drawStart = { x: 0, y: 0 };
    this.tempBox = null;

    // Visual Palette Cache
    this.classColors = {};
    this.colorPalette = [
      '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
      '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#14b8a6'
    ];

    // Callbacks
    this.onBoxChange = null;
    this.onCursorMove = null;
    this.onSnapRequest = null;
    this.onClickRequest = null;

    this.initEventListeners();
    this.resizeCanvas();
  }

  resizeCanvas() {
    const rect = this.viewport.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.render();
  }

  loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.image = img;
        this.imageWidth = img.naturalWidth;
        this.imageHeight = img.naturalHeight;
        this.fitToScreen();
        resolve(img);
      };
      img.onerror = reject;
      img.src = src;
    });
  }

  fitToScreen() {
    if (!this.image) return;
    const vpRect = this.viewport.getBoundingClientRect();
    const pad = 40;
    const scaleX = (vpRect.width - pad) / this.imageWidth;
    const scaleY = (vpRect.height - pad) / this.imageHeight;
    this.scale = Math.min(scaleX, scaleY, 1.5);
    this.offsetX = (vpRect.width - this.imageWidth * this.scale) / 2;
    this.offsetY = (vpRect.height - this.imageHeight * this.scale) / 2;
    this.render();
  }

  screenToImageCoords(screenX, screenY) {
    const rect = this.viewport.getBoundingClientRect();
    const localX = screenX - rect.left;
    const localY = screenY - rect.top;
    const imgX = (localX - this.offsetX) / this.scale;
    const imgY = (localY - this.offsetY) / this.scale;
    return {
      x: Math.max(0, Math.min(this.imageWidth, imgX)),
      y: Math.max(0, Math.min(this.imageHeight, imgY))
    };
  }

  imageToScreenCoords(imgX, imgY) {
    return {
      x: this.offsetX + imgX * this.scale,
      y: this.offsetY + imgY * this.scale
    };
  }

  getColorForClass(className, classId = 0) {
    if (!this.classColors[className]) {
      const idx = Math.abs(classId) % this.colorPalette.length;
      this.classColors[className] = this.colorPalette[idx];
    }
    return this.classColors[className];
  }

  saveState() {
    this.history.push(JSON.parse(JSON.stringify(this.boxes)));
    if (this.history.length > 30) this.history.shift();
  }

  undo() {
    if (this.history.length > 0) {
      this.boxes = this.history.pop();
      this.selectedBoxIndex = -1;
      this.render();
      if (this.onBoxChange) this.onBoxChange(this.boxes);
    }
  }

  addBox(box) {
    this.saveState();
    this.boxes.push(box);
    this.selectedBoxIndex = this.boxes.length - 1;
    this.render();
    if (this.onBoxChange) this.onBoxChange(this.boxes);
  }

  deleteSelectedBox() {
    if (this.selectedBoxIndex >= 0 && this.selectedBoxIndex < this.boxes.length) {
      this.saveState();
      this.boxes.splice(this.selectedBoxIndex, 1);
      this.selectedBoxIndex = -1;
      this.render();
      if (this.onBoxChange) this.onBoxChange(this.boxes);
    }
  }

  getHandleAt(screenX, screenY) {
    if (this.selectedBoxIndex < 0) return null;
    const b = this.boxes[this.selectedBoxIndex];
    const tl = this.imageToScreenCoords(b.xmin, b.ymin);
    const br = this.imageToScreenCoords(b.xmax, b.ymax);
    const w = br.x - tl.x;
    const h = br.y - tl.y;
    const hs = 8; // handle size

    const handles = {
      tl: { x: tl.x, y: tl.y },
      tr: { x: br.x, y: tl.y },
      bl: { x: tl.x, y: br.y },
      br: { x: br.x, y: br.y },
      tc: { x: tl.x + w / 2, y: tl.y },
      bc: { x: tl.x + w / 2, y: br.y },
      lc: { x: tl.x, y: tl.y + h / 2 },
      rc: { x: br.x, y: tl.y + h / 2 }
    };

    for (const [key, pt] of Object.entries(handles)) {
      if (Math.hypot(screenX - pt.x, screenY - pt.y) <= hs + 2) {
        return key;
      }
    }
    return null;
  }

  getBoxAt(imgX, imgY) {
    // Check in reverse order so topmost box is selected first
    for (let i = this.boxes.length - 1; i >= 0; i--) {
      const b = this.boxes[i];
      if (imgX >= b.xmin && imgX <= b.xmax && imgY >= b.ymin && imgY <= b.ymax) {
        return i;
      }
    }
    return -1;
  }

  initEventListeners() {
    window.addEventListener('resize', () => this.resizeCanvas());

    this.viewport.addEventListener('mousedown', (e) => this.handleMouseDown(e));
    window.addEventListener('mousemove', (e) => this.handleMouseMove(e));
    window.addEventListener('mouseup', (e) => this.handleMouseUp(e));
    this.viewport.addEventListener('wheel', (e) => this.handleWheel(e), { passive: false });

    // Prevent default right-click context menu
    this.viewport.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  handleMouseDown(e) {
    if (!this.image) return;
    const rect = this.viewport.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    const { x: imgX, y: imgY } = this.screenToImageCoords(e.clientX, e.clientY);

    // Pan mode check (Middle click, Space+Click, or right-click)
    if (e.button === 1 || e.button === 2 || e.spaceKey) {
      this.isPanning = true;
      this.dragStart = { x: e.clientX - this.offsetX, y: e.clientY - this.offsetY };
      this.viewport.style.cursor = 'grab';
      return;
    }

    if (e.button !== 0) return; // Only primary click
    this.isMouseDown = true;

    // 1. Check if clicking on an active resize handle
    const handle = this.getHandleAt(screenX, screenY);
    if (handle) {
      this.isResizing = true;
      this.activeHandle = handle;
      this.saveState();
      return;
    }

    // 2. Click-to-box tool
    if (this.currentTool === 'click') {
      if (this.onClickRequest) {
        this.onClickRequest(Math.round(imgX), Math.round(imgY));
      }
      return;
    }

    // 3. Check if clicking inside an existing box (selection or drag move)
    const clickedBoxIdx = this.getBoxAt(imgX, imgY);
    if (clickedBoxIdx >= 0 && this.currentTool === 'manual') {
      this.selectedBoxIndex = clickedBoxIdx;
      this.isDraggingBox = true;
      this.dragStart = { x: imgX, y: imgY };
      this.saveState();
      this.render();
      if (this.onBoxChange) this.onBoxChange(this.boxes);
      return;
    }

    // 4. Starting a new bounding box (Manual or Smart Snap)
    this.selectedBoxIndex = -1;
    this.isDrawing = true;
    this.drawStart = { x: imgX, y: imgY };
    this.tempBox = {
      xmin: imgX,
      ymin: imgY,
      xmax: imgX,
      ymax: imgY
    };
    this.render();
  }

  handleMouseMove(e) {
    const rect = this.viewport.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;
    const { x: imgX, y: imgY } = this.screenToImageCoords(e.clientX, e.clientY);

    this.currentMouse = { imgX, imgY, screenX, screenY };
    if (this.onCursorMove) {
      this.onCursorMove(Math.round(imgX), Math.round(imgY), this.tempBox);
    }

    // Panning
    if (this.isPanning) {
      this.offsetX = e.clientX - this.dragStart.x;
      this.offsetY = e.clientY - this.dragStart.y;
      this.render();
      return;
    }

    // Resizing
    if (this.isResizing && this.selectedBoxIndex >= 0) {
      const b = this.boxes[this.selectedBoxIndex];
      switch (this.activeHandle) {
        case 'tl': b.xmin = imgX; b.ymin = imgY; break;
        case 'tr': b.xmax = imgX; b.ymin = imgY; break;
        case 'bl': b.xmin = imgX; b.ymax = imgY; break;
        case 'br': b.xmax = imgX; b.ymax = imgY; break;
        case 'tc': b.ymin = imgY; break;
        case 'bc': b.ymax = imgY; break;
        case 'lc': b.xmin = imgX; break;
        case 'rc': b.xmax = imgX; break;
      }
      // Ensure min/max invariant
      if (b.xmin > b.xmax) [b.xmin, b.xmax] = [b.xmax, b.xmin];
      if (b.ymin > b.ymax) [b.ymin, b.ymax] = [b.ymax, b.ymin];
      this.render();
      return;
    }

    // Dragging an existing box
    if (this.isDraggingBox && this.selectedBoxIndex >= 0) {
      const b = this.boxes[this.selectedBoxIndex];
      const dx = imgX - this.dragStart.x;
      const dy = imgY - this.dragStart.y;
      b.xmin += dx;
      b.xmax += dx;
      b.ymin += dy;
      b.ymax += dy;
      this.dragStart = { x: imgX, y: imgY };
      this.render();
      return;
    }

    // Drawing a new box
    if (this.isDrawing && this.tempBox) {
      this.tempBox.xmin = Math.min(this.drawStart.x, imgX);
      this.tempBox.ymin = Math.min(this.drawStart.y, imgY);
      this.tempBox.xmax = Math.max(this.drawStart.x, imgX);
      this.tempBox.ymax = Math.max(this.drawStart.y, imgY);
      this.render();
      return;
    }

    // Update cursor style for hover
    if (this.image) {
      const handle = this.getHandleAt(screenX, screenY);
      if (handle) {
        const cursors = {
          tl: 'nwse-resize', br: 'nwse-resize',
          tr: 'nesw-resize', bl: 'nesw-resize',
          tc: 'ns-resize', bc: 'ns-resize',
          lc: 'ew-resize', rc: 'ew-resize'
        };
        this.viewport.style.cursor = cursors[handle] || 'default';
      } else if (this.getBoxAt(imgX, imgY) >= 0) {
        this.viewport.style.cursor = 'move';
      } else {
        this.viewport.style.cursor = this.currentTool === 'click' ? 'pointer' : 'crosshair';
      }
    }

    if (this.showCrosshairs) {
      this.render();
    }
  }

  handleMouseUp(e) {
    if (this.isPanning) {
      this.isPanning = false;
      this.viewport.style.cursor = 'default';
      return;
    }

    if (this.isResizing || this.isDraggingBox) {
      this.isResizing = false;
      this.isDraggingBox = false;
      this.activeHandle = null;
      if (this.onBoxChange) this.onBoxChange(this.boxes);
      return;
    }

    if (this.isDrawing && this.tempBox) {
      this.isDrawing = false;
      const bw = this.tempBox.xmax - this.tempBox.xmin;
      const bh = this.tempBox.ymax - this.tempBox.ymin;

      if (bw > 6 && bh > 6) {
        if (this.currentTool === 'snap' && this.onSnapRequest) {
          // Send rough box to OpenCV contour snap engine
          this.onSnapRequest([this.tempBox.xmin, this.tempBox.ymin, this.tempBox.xmax, this.tempBox.ymax]);
        } else {
          // Commit manual box
          this.commitNewBox(this.tempBox.xmin, this.tempBox.ymin, this.tempBox.xmax, this.tempBox.ymax);
        }
      }
      this.tempBox = null;
      this.render();
    }

    this.isMouseDown = false;
  }

  handleWheel(e) {
    e.preventDefault();
    if (!this.image) return;

    const zoomFactor = 1.12;
    const direction = e.deltaY < 0 ? 1 : -1;
    const factor = direction > 0 ? zoomFactor : 1 / zoomFactor;

    const rect = this.viewport.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const newScale = Math.max(0.1, Math.min(10.0, this.scale * factor));
    // Zoom toward mouse pointer
    this.offsetX = mouseX - (mouseX - this.offsetX) * (newScale / this.scale);
    this.offsetY = mouseY - (mouseY - this.offsetY) * (newScale / this.scale);
    this.scale = newScale;

    this.render();
    if (this.onZoomChange) this.onZoomChange(this.scale);
  }

  commitNewBox(xmin, ymin, xmax, ymax, className = null, classId = null) {
    const defaultClass = className || (window.activeClass ? window.activeClass.name : 'object');
    const defaultId = classId !== null ? classId : (window.activeClass ? window.activeClass.id : 0);

    const newBox = {
      class_name: defaultClass,
      class_id: defaultId,
      xmin: Math.round(xmin),
      ymin: Math.round(ymin),
      xmax: Math.round(xmax),
      ymax: Math.round(ymax),
      conf: 1.0
    };
    this.addBox(newBox);
  }

  render() {
    const ctx = this.ctx;
    const rect = this.viewport.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);

    if (!this.image) {
      ctx.fillStyle = '#64748b';
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Select or upload an image to begin labeling', rect.width / 2, rect.height / 2);
      return;
    }

    // 1. Draw Image
    ctx.save();
    ctx.imageSmoothingEnabled = this.scale < 2.0;
    ctx.drawImage(
      this.image,
      this.offsetX,
      this.offsetY,
      this.imageWidth * this.scale,
      this.imageHeight * this.scale
    );
    ctx.restore();

    // 2. Draw Committed Bounding Boxes
    this.boxes.forEach((b, idx) => {
      const isSelected = idx === this.selectedBoxIndex;
      const color = this.getColorForClass(b.class_name, b.class_id);

      const tl = this.imageToScreenCoords(b.xmin, b.ymin);
      const br = this.imageToScreenCoords(b.xmax, b.ymax);
      const w = br.x - tl.x;
      const h = br.y - tl.y;

      // Box fill & border
      ctx.fillStyle = isSelected ? 'rgba(59, 130, 246, 0.25)' : `${color}22`;
      ctx.fillRect(tl.x, tl.y, w, h);

      ctx.strokeStyle = isSelected ? '#ffffff' : color;
      ctx.lineWidth = isSelected ? 2.5 : 1.8;
      ctx.strokeRect(tl.x, tl.y, w, h);

      // Label Pill
      const labelText = `${b.class_name}${b.conf < 1.0 ? ` (${Math.round(b.conf * 100)}%)` : ''}`;
      ctx.font = '600 11px Inter, sans-serif';
      const textWidth = ctx.measureText(labelText).width;
      const pillHeight = 18;
      const pillY = Math.max(0, tl.y - pillHeight);

      ctx.fillStyle = isSelected ? '#3b82f6' : color;
      ctx.fillRect(tl.x, pillY, textWidth + 12, pillHeight);

      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, tl.x + 6, pillY + 13);

      // Draw 8 Resize Handles if selected
      if (isSelected) {
        this.drawHandles(tl.x, tl.y, w, h);
      }
    });

    // 3. Draw Active Drawing Temp Box
    if (this.isDrawing && this.tempBox) {
      const tl = this.imageToScreenCoords(this.tempBox.xmin, this.tempBox.ymin);
      const br = this.imageToScreenCoords(this.tempBox.xmax, this.tempBox.ymax);
      const w = br.x - tl.x;
      const h = br.y - tl.y;

      ctx.fillStyle = this.currentTool === 'snap' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(59, 130, 246, 0.2)';
      ctx.fillRect(tl.x, tl.y, w, h);

      ctx.strokeStyle = this.currentTool === 'snap' ? '#10b981' : '#3b82f6';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(tl.x, tl.y, w, h);
      ctx.setLineDash([]);
    }

    // 4. Draw Crosshair Guides
    if (this.showCrosshairs && this.currentMouse.screenX >= 0 && this.currentMouse.screenY >= 0) {
      const sx = this.currentMouse.screenX;
      const sy = this.currentMouse.screenY;

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 4]);

      ctx.beginPath();
      ctx.moveTo(sx, 0);
      ctx.lineTo(sx, rect.height);
      ctx.moveTo(0, sy);
      ctx.lineTo(rect.width, sy);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  drawHandles(x, y, w, h) {
    const ctx = this.ctx;
    const hs = 6;
    const pts = [
      { x: x, y: y }, { x: x + w / 2, y: y }, { x: x + w, y: y },
      { x: x, y: y + h / 2 }, { x: x + w, y: y + h / 2 },
      { x: x, y: y + h }, { x: x + w / 2, y: y + h }, { x: x + w, y: y + h }
    ];

    pts.forEach(p => {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(p.x - hs / 2, p.y - hs / 2, hs, hs);
      ctx.strokeStyle = '#0f1117';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(p.x - hs / 2, p.y - hs / 2, hs, hs);
    });
  }
}
