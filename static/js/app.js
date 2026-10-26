/**
 * detect-labeler: Main Application Controller
 * ===========================================
 * Connects Canvas engine, REST APIs, Dynamic Taxonomy Editor,
 * Multi-domain presets, and Keyboard Shortcuts.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Application State
  const state = {
    currentImage: null,
    images: [],
    taxonomies: [],
    classes: ['object', 'person', 'vehicle', 'animal', 'equipment'],
    activeClassIndex: 0,
    exportFormat: 'yolo',
    currentTool: 'manual', // 'manual', 'snap', 'click'
    snapPadding: 0.20,
    clickRadius: 150,
    maxProposals: 15
  };

  // DOM Elements
  const canvasEl = document.getElementById('annotationCanvas');
  const viewportEl = document.getElementById('canvasViewport');
  const presetSelect = document.getElementById('presetSelect');
  const exportFormatSelect = document.getElementById('exportFormat');
  const btnSave = document.getElementById('btnSave');
  const btnExportZip = document.getElementById('btnExportZip');
  const classPaletteList = document.getElementById('classPaletteList');
  const newClassInput = document.getElementById('newClassInput');
  const btnAddClass = document.getElementById('btnAddClass');
  const classCountBadge = document.getElementById('classCountBadge');
  const boxesList = document.getElementById('boxesList');
  const boxCountLabel = document.getElementById('boxCountLabel');
  const imageCountLabel = document.getElementById('imageCountLabel');
  const imagesFilmstrip = document.getElementById('imagesFilmstrip');
  const statusToast = document.getElementById('statusToast');

  // Tool Buttons
  const toolButtons = {
    manual: document.getElementById('toolManual'),
    snap: document.getElementById('toolSnap'),
    click: document.getElementById('toolClick'),
    propose: document.getElementById('toolPropose')
  };

  // HUD Elements
  const hudActiveTool = document.getElementById('hudActiveTool');
  const hudCoords = document.getElementById('hudCoords');
  const hudBoxDims = document.getElementById('hudBoxDims');
  const hudZoom = document.getElementById('hudZoom');
  const btnZoomIn = document.getElementById('btnZoomIn');
  const btnZoomOut = document.getElementById('btnZoomOut');
  const btnZoomFit = document.getElementById('btnZoomFit');
  const btnCrosshairToggle = document.getElementById('btnCrosshairToggle');

  // Sliders
  const snapPaddingInput = document.getElementById('snapPaddingInput');
  const paddingVal = document.getElementById('paddingVal');
  const clickRadiusInput = document.getElementById('clickRadiusInput');
  const radiusVal = document.getElementById('radiusVal');
  const maxProposalsInput = document.getElementById('maxProposalsInput');
  const proposalsVal = document.getElementById('proposalsVal');

  // File Inputs
  const imageUploadInput = document.getElementById('imageUploadInput');
  const taxonomyFileInput = document.getElementById('taxonomyFileInput');
  const btnExportTaxonomy = document.getElementById('btnExportTaxonomy');

  // Initialize Canvas
  const annotator = new AnnotationCanvas(canvasEl, viewportEl);

  function showToast(message, isError = false) {
    statusToast.textContent = message;
    statusToast.style.backgroundColor = isError ? 'rgba(239, 68, 68, 0.9)' : 'rgba(16, 185, 129, 0.9)';
    statusToast.classList.remove('hidden');
    setTimeout(() => {
      statusToast.classList.add('hidden');
    }, 2500);
  }

  function getActiveClass() {
    return {
      name: state.classes[state.activeClassIndex] || 'object',
      id: state.activeClassIndex
    };
  }

  window.activeClass = getActiveClass();

  function setActiveClass(idx) {
    if (idx >= 0 && idx < state.classes.length) {
      state.activeClassIndex = idx;
      window.activeClass = getActiveClass();
      renderTaxonomyPalette();
    }
  }

  function setTool(toolName) {
    state.currentTool = toolName;
    annotator.currentTool = toolName;
    Object.keys(toolButtons).forEach(t => {
      if (toolButtons[t]) {
        toolButtons[t].classList.toggle('active', t === toolName);
      }
    });
    hudActiveTool.innerHTML = `Tool: <strong>${toolName.toUpperCase()}</strong>`;
  }

  // Render Taxonomy Class Palette
  function renderTaxonomyPalette() {
    classPaletteList.innerHTML = '';
    classCountBadge.textContent = `${state.classes.length} classes`;

    state.classes.forEach((cname, idx) => {
      const color = annotator.getColorForClass(cname, idx);
      const item = document.createElement('div');
      item.className = `class-item ${idx === state.activeClassIndex ? 'active' : ''}`;
      item.innerHTML = `
        <div class="class-meta">
          <span class="class-color-dot" style="background-color: ${color}"></span>
          <span class="class-name">${cname}</span>
        </div>
        <span class="class-key">${idx < 9 ? idx + 1 : ''}</span>
      `;
      item.addEventListener('click', () => setActiveClass(idx));
      classPaletteList.appendChild(item);
    });
  }

  // Render Box Inspector List
  function renderBoxesList(boxes) {
    boxesList.innerHTML = '';
    boxCountLabel.textContent = boxes.length;

    if (boxes.length === 0) {
      boxesList.innerHTML = `<p style="font-size: 11px; color: var(--text-dim); text-align: center; padding: 20px 0;">No annotations on this frame.</p>`;
      return;
    }

    boxes.forEach((b, idx) => {
      const isSelected = idx === annotator.selectedBoxIndex;
      const color = annotator.getColorForClass(b.class_name, b.class_id);
      const card = document.createElement('div');
      card.className = `box-card ${isSelected ? 'selected' : ''}`;
      card.innerHTML = `
        <div class="box-card-left">
          <span class="class-color-dot" style="background-color: ${color}"></span>
          <div>
            <div style="font-size: 11px; font-weight: 600;">${b.class_name}</div>
            <div class="box-coords">${Math.round(b.xmin)}, ${Math.round(b.ymin)} → ${Math.round(b.xmax)}, ${Math.round(b.ymax)}</div>
          </div>
        </div>
        <button class="box-delete-btn" title="Delete box">&times;</button>
      `;

      card.addEventListener('click', (e) => {
        if (!e.target.classList.contains('box-delete-btn')) {
          annotator.selectedBoxIndex = idx;
          annotator.render();
          renderBoxesList(boxes);
        }
      });

      card.querySelector('.box-delete-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        annotator.selectedBoxIndex = idx;
        annotator.deleteSelectedBox();
      });

      boxesList.appendChild(card);
    });
  }

  // Render Images Filmstrip
  function renderImagesFilmstrip() {
    imagesFilmstrip.innerHTML = '';
    imageCountLabel.textContent = state.images.length;

    state.images.forEach(img => {
      const isActive = img.filename === state.currentImage;
      const item = document.createElement('div');
      item.className = `filmstrip-item ${isActive ? 'active' : ''}`;
      item.innerHTML = `
        <img class="filmstrip-thumb" src="/api/image/${img.filename}" alt="${img.filename}" loading="lazy" />
        <div class="filmstrip-info">
          <div class="filmstrip-name">${img.filename}</div>
          <div class="filmstrip-meta">${img.has_annotations ? `✓ ${img.box_count} boxes` : 'Unlabeled'}</div>
        </div>
      `;
      item.addEventListener('click', () => loadImage(img.filename));
      imagesFilmstrip.appendChild(item);
    });
  }

  // Load Image and its Annotations
  async function loadImage(filename) {
    try {
      state.currentImage = filename;
      await annotator.loadImage(`/api/image/${filename}`);

      // Fetch annotations
      const res = await fetch(`/api/annotations/${filename}`);
      if (res.ok) {
        const data = await res.json();
        annotator.boxes = data.boxes.map(b => ({
          class_name: b.class_name,
          class_id: b.class_id,
          xmin: b.xmin,
          ymin: b.ymin,
          xmax: b.xmax,
          ymax: b.ymax,
          conf: b.conf || 1.0
        }));
        annotator.selectedBoxIndex = -1;
        annotator.render();
        renderBoxesList(annotator.boxes);
      }
      renderImagesFilmstrip();
    } catch (err) {
      showToast(`Error loading image: ${err.message}`, true);
    }
  }

  // Save Annotations
  async function saveAnnotations() {
    if (!state.currentImage) return;
    try {
      const payload = {
        format: state.exportFormat,
        boxes: annotator.boxes,
        img_w: annotator.imageWidth,
        img_h: annotator.imageHeight
      };

      const res = await fetch(`/api/annotations/${state.currentImage}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        showToast(`Saved ${annotator.boxes.length} annotations (${state.exportFormat.toUpperCase()})`);
        // Refresh image metadata in filmstrip
        const imgObj = state.images.find(i => i.filename === state.currentImage);
        if (imgObj) {
          imgObj.has_annotations = annotator.boxes.length > 0;
          imgObj.box_count = annotator.boxes.length;
          renderImagesFilmstrip();
        }
      } else {
        showToast('Failed to save annotations', true);
      }
    } catch (err) {
      showToast(`Error saving: ${err.message}`, true);
    }
  }

  // Smart Snap Handler
  annotator.onSnapRequest = async (roughBox) => {
    if (!state.currentImage) return;
    try {
      showToast('⚡ Snapping contour...');
      const res = await fetch('/api/snap_box', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_name: state.currentImage,
          rough_box: roughBox,
          padding: state.snapPadding
        })
      });

      if (res.ok) {
        const data = await res.json();
        const [x1, y1, x2, y2] = data.snapped_box;
        const ac = getActiveClass();
        annotator.commitNewBox(x1, y1, x2, y2, ac.name, ac.id);
        showToast('✓ Snapped to object contour!');
      }
    } catch (err) {
      showToast('Contour snap failed', true);
    }
  };

  // Click-to-Box Handler
  annotator.onClickRequest = async (clickX, clickY) => {
    if (!state.currentImage) return;
    try {
      showToast('👆 Segmenting object...');
      const res = await fetch('/api/click_box', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_name: state.currentImage,
          click_x: clickX,
          click_y: clickY,
          radius: state.clickRadius
        })
      });

      if (res.ok) {
        const data = await res.json();
        const [x1, y1, x2, y2] = data.box;
        const ac = getActiveClass();
        annotator.commitNewBox(x1, y1, x2, y2, ac.name, ac.id);
        showToast('✓ Click-to-box generated!');
      }
    } catch (err) {
      showToast('Click-to-box failed', true);
    }
  };

  // Salient Auto-Proposals
  toolButtons.propose.addEventListener('click', async () => {
    if (!state.currentImage) {
      showToast('Select an image first', true);
      return;
    }
    try {
      showToast('✨ Extracting salient proposals...');
      const res = await fetch('/api/salient_proposals', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_name: state.currentImage,
          max_proposals: state.maxProposals
        })
      });

      if (res.ok) {
        const data = await res.json();
        annotator.saveState();
        const ac = getActiveClass();
        data.proposals.forEach(p => {
          annotator.boxes.push({
            class_name: ac.name,
            class_id: ac.id,
            xmin: p[0],
            ymin: p[1],
            xmax: p[2],
            ymax: p[3],
            conf: 0.85
          });
        });
        annotator.render();
        renderBoxesList(annotator.boxes);
        showToast(`✓ Generated ${data.proposals.length} candidate boxes!`);
      }
    } catch (err) {
      showToast('Auto-propose failed', true);
    }
  });

  // Callbacks from Canvas
  annotator.onBoxChange = (boxes) => {
    renderBoxesList(boxes);
  };

  annotator.onCursorMove = (imgX, imgY, tempBox) => {
    hudCoords.textContent = `X: ${imgX}, Y: ${imgY}`;
    if (tempBox) {
      const w = Math.round(tempBox.xmax - tempBox.xmin);
      const h = Math.round(tempBox.ymax - tempBox.ymin);
      hudBoxDims.textContent = `Size: ${w} × ${h}`;
    } else if (annotator.selectedBoxIndex >= 0) {
      const b = annotator.boxes[annotator.selectedBoxIndex];
      const w = Math.round(b.xmax - b.xmin);
      const h = Math.round(b.ymax - b.ymin);
      hudBoxDims.textContent = `Size: ${w} × ${h}`;
    } else {
      hudBoxDims.textContent = `Size: 0 × 0`;
    }
  };

  annotator.onZoomChange = (scale) => {
    hudZoom.textContent = `${Math.round(scale * 100)}%`;
  };

  // Tool Mode Clicks
  toolButtons.manual.addEventListener('click', () => setTool('manual'));
  toolButtons.snap.addEventListener('click', () => setTool('snap'));
  toolButtons.click.addEventListener('click', () => setTool('click'));

  // HUD Controls
  btnZoomIn.addEventListener('click', () => {
    annotator.scale = Math.min(10.0, annotator.scale * 1.25);
    annotator.render();
    annotator.onZoomChange(annotator.scale);
  });

  btnZoomOut.addEventListener('click', () => {
    annotator.scale = Math.max(0.1, annotator.scale / 1.25);
    annotator.render();
    annotator.onZoomChange(annotator.scale);
  });

  btnZoomFit.addEventListener('click', () => annotator.fitToScreen());

  btnCrosshairToggle.addEventListener('click', () => {
    annotator.showCrosshairs = !annotator.showCrosshairs;
    btnCrosshairToggle.classList.toggle('active', annotator.showCrosshairs);
    annotator.render();
  });

  // Save & Export Buttons
  btnSave.addEventListener('click', saveAnnotations);
  exportFormatSelect.addEventListener('change', (e) => {
    state.exportFormat = e.target.value;
  });

  btnExportZip.addEventListener('click', () => {
    window.location.href = `/api/export_dataset?format=${state.exportFormat}`;
  });

  // Taxonomy Management
  btnAddClass.addEventListener('click', () => {
    const val = newClassInput.value.trim().toLowerCase();
    if (val && !state.classes.includes(val)) {
      state.classes.push(val);
      newClassInput.value = '';
      setActiveClass(state.classes.length - 1);
      renderTaxonomyPalette();
      showToast(`Added class "${val}"`);
    }
  });

  newClassInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') btnAddClass.click();
  });

  presetSelect.addEventListener('change', async (e) => {
    const selectedId = e.target.value;
    const preset = state.taxonomies.find(t => t.id === selectedId);
    if (preset && preset.classes && preset.classes.length > 0) {
      state.classes = [...preset.classes];
      state.activeClassIndex = 0;
      window.activeClass = getActiveClass();
      renderTaxonomyPalette();
      showToast(`Loaded ${preset.name} preset`);
    }
  });

  btnExportTaxonomy.addEventListener('click', () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify({ classes: state.classes }, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute("href", dataStr);
    dlAnchor.setAttribute("download", "taxonomy.json");
    dlAnchor.click();
  });

  taxonomyFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      try {
        if (file.name.endsWith('.json')) {
          const parsed = JSON.parse(text);
          state.classes = parsed.classes || parsed.names || Object.values(parsed);
        } else {
          // txt or yaml list
          state.classes = text.split(/\r?\n/).map(l => l.trim()).filter(l => l && !l.startsWith('#'));
        }
        state.activeClassIndex = 0;
        window.activeClass = getActiveClass();
        renderTaxonomyPalette();
        showToast(`Imported ${state.classes.length} classes`);
      } catch (err) {
        showToast('Failed to parse taxonomy file', true);
      }
    };
    reader.readAsText(file);
  });

  // Sliders
  snapPaddingInput.addEventListener('input', (e) => {
    state.snapPadding = parseInt(e.target.value) / 100.0;
    paddingVal.textContent = `${e.target.value}%`;
  });

  clickRadiusInput.addEventListener('input', (e) => {
    state.clickRadius = parseInt(e.target.value);
    radiusVal.textContent = `${e.target.value} px`;
  });

  maxProposalsInput.addEventListener('input', (e) => {
    state.maxProposals = parseInt(e.target.value);
    proposalsVal.textContent = e.target.value;
  });

  // Image Upload
  imageUploadInput.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    for (const f of files) {
      const formData = new FormData();
      formData.append('file', f);
      try {
        await fetch('/api/upload_image', { method: 'POST', body: formData });
      } catch (err) {
        console.error(err);
      }
    }
    await loadInitialData();
    showToast(`Uploaded ${files.length} images`);
  });

  // Tabs
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      document.getElementById(targetId).classList.add('active');
    });
  });

  // Global Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    // Ignore keystrokes when typing into text inputs
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

    // 1-9 Class switching
    const num = parseInt(e.key);
    if (!isNaN(num) && num >= 1 && num <= 9) {
      setActiveClass(num - 1);
      return;
    }

    if (e.key === 'b' || e.key === 'B') {
      setTool('manual');
    } else if (e.key === 's' || e.key === 'S') {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        saveAnnotations();
      } else {
        setTool('snap');
      }
    } else if (e.key === 'c' || e.key === 'C') {
      setTool('click');
    } else if (e.key === 'a' || e.key === 'A') {
      toolButtons.propose.click();
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      annotator.deleteSelectedBox();
    } else if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {
      e.preventDefault();
      annotator.undo();
    }
  });

  // Initial Data Fetch
  async function loadInitialData() {
    try {
      // 1. Fetch Taxonomies
      const taxRes = await fetch('/api/taxonomies');
      if (taxRes.ok) {
        const taxData = await taxRes.json();
        state.taxonomies = taxData.presets || [];
        presetSelect.innerHTML = '<option value="">-- Load Domain Preset --</option>';
        state.taxonomies.forEach(t => {
          const opt = document.createElement('option');
          opt.value = t.id;
          opt.textContent = `${t.name} (${t.classes.length} classes)`;
          presetSelect.appendChild(opt);
        });
      }

      // 2. Fetch Images
      const imgRes = await fetch('/api/images');
      if (imgRes.ok) {
        const imgData = await imgRes.json();
        state.images = imgData.images || [];
        renderImagesFilmstrip();

        // Load first image if present
        if (state.images.length > 0 && !state.currentImage) {
          await loadImage(state.images[0].filename);
        }
      }

      renderTaxonomyPalette();
    } catch (err) {
      console.error(err);
    }
  }

  loadInitialData();
});
