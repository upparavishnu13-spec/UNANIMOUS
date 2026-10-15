/* ============================================================
   CropSense AI — app.js
   Vanilla JavaScript (no frameworks, no jQuery, no Chart.js)
   Canvas 2D API used for all charts
   ============================================================ */

'use strict';

/* ── 1. Constants & Config ─────────────────────────────────── */
const API_BASE = window.location.origin;

let currentLang = 'en';

const CROP_EMOJIS = {
  tomato:  '🍅',
  potato:  '🥔',
  corn:    '🌽',
  apple:   '🍎',
  grape:   '🍇',
  wheat:   '🌾',
  rice:    '🍚',
  pepper:  '🌶️',
  default: '🌿',
};

const SAMPLE_IMAGES = [
  { name: 'tomato_early_blight.jpg',  url: '/static/samples/tomato_early_blight.jpg',  label: 'Tomato Early Blight' },
  { name: 'potato_late_blight.jpg',   url: '/static/samples/potato_late_blight.jpg',   label: 'Potato Late Blight'  },
  { name: 'corn_common_rust.jpg',     url: '/static/samples/corn_common_rust.jpg',      label: 'Corn Common Rust'   },
  { name: 'apple_scab.jpg',           url: '/static/samples/apple_scab.jpg',            label: 'Apple Scab'         },
  { name: 'tomato_healthy.jpg',       url: '/static/samples/tomato_healthy.jpg',        label: 'Healthy Tomato'     },
];

/* ── 2. Language System ────────────────────────────────────── */
/**
 * Switch the UI language between English and Telugu.
 * Any element with data-en / data-te attributes gets updated.
 */
function setLanguage(lang) {
  currentLang = lang;

  document.querySelectorAll('[data-en]').forEach(el => {
    const text = el.getAttribute(`data-${lang}`);
    if (text !== null) el.textContent = text;
  });

  // Update lang-btn active state
  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.lang === lang);
  });

  // Re-render results if visible
  const resultsPanel = document.getElementById('resultsPanel');
  if (resultsPanel && !resultsPanel.classList.contains('hidden') && window._lastPrediction) {
    renderResults(window._lastPrediction);
  }
}

/* ── 3. Toast Notifications ────────────────────────────────── */
function showToast(message, type = 'success', duration = 4000) {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${icons[type] || icons.info}</span><span>${message}</span>`;

  container.appendChild(toast);

  const removeToast = () => {
    toast.style.animation = 'toast-out 0.3s ease forwards';
    setTimeout(() => toast.remove(), 300);
  };

  setTimeout(removeToast, duration);
  toast.addEventListener('click', removeToast);
}

/* ── 4. Navigation ─────────────────────────────────────────── */
function initNavigation() {
  const navbar     = document.getElementById('navbar');
  const hamburger  = document.getElementById('hamburger');
  const navLinks   = document.getElementById('navLinks');
  const allLinks   = document.querySelectorAll('.nav-links a[href^="#"]');

  // Sticky shadow + scrolled class
  window.addEventListener('scroll', () => {
    if (navbar) navbar.classList.toggle('scrolled', window.scrollY > 10);
  }, { passive: true });

  // Hamburger toggle
  if (hamburger && navLinks) {
    hamburger.addEventListener('click', () => {
      hamburger.classList.toggle('open');
      navLinks.classList.toggle('mobile-open');
    });

    // Close on link click (mobile)
    navLinks.querySelectorAll('a').forEach(a => {
      a.addEventListener('click', () => {
        hamburger.classList.remove('open');
        navLinks.classList.remove('mobile-open');
      });
    });
  }

  // Active link highlight using IntersectionObserver
  const sections = document.querySelectorAll('section[id]');

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        allLinks.forEach(link => {
          link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
        });
      }
    });
  }, { rootMargin: '-40% 0px -55% 0px' });

  sections.forEach(sec => observer.observe(sec));

  // Language buttons
  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.addEventListener('click', () => setLanguage(btn.dataset.lang));
  });
}

/* ── 5. Upload & Drag-and-Drop ─────────────────────────────── */
let selectedFile = null;

function initUpload() {
  const uploadArea    = document.getElementById('uploadArea');
  const fileInput     = document.getElementById('fileInput');
  const previewCard   = document.getElementById('previewCard');
  const previewImage  = document.getElementById('previewImage');
  const fileNameEl    = document.getElementById('previewFileName');
  const fileSizeEl    = document.getElementById('previewFileSize');
  const clearBtn      = document.getElementById('clearBtn');
  const analyzeBtn    = document.getElementById('analyzeBtn');

  if (!uploadArea || !fileInput) return;

  // Click to browse
  uploadArea.addEventListener('click', (e) => {
    if (e.target.closest('#sampleGrid') || e.target.closest('#cameraContainer')) return;
    fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  // Drag-and-drop
  uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
  });

  uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
  });

  uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  // Clear
  if (clearBtn) clearBtn.addEventListener('click', clearImage);

  // Analyze
  if (analyzeBtn) analyzeBtn.addEventListener('click', analyzeImage);
}

function handleFile(file) {
  // Validate type
  const allowed = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp', 'image/gif', 'image/bmp'];
  if (!allowed.includes(file.type)) {
    showToast('Invalid file type. Please use JPG, PNG, WEBP, or BMP.', 'error');
    return;
  }

  // Validate size (10 MB max)
  const maxMB = 10;
  if (file.size > maxMB * 1024 * 1024) {
    showToast(`File too large. Maximum size is ${maxMB} MB.`, 'error');
    return;
  }

  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    const previewImage = document.getElementById('previewImage');
    const fileNameEl   = document.getElementById('previewFileName');
    const fileSizeEl   = document.getElementById('previewFileSize');
    const previewCard  = document.getElementById('previewCard');

    if (previewImage)  previewImage.src = e.target.result;
    if (fileNameEl)    fileNameEl.textContent = file.name;
    if (fileSizeEl)    fileSizeEl.textContent = formatFileSize(file.size);
    if (previewCard)   previewCard.classList.remove('hidden');

    // Hide previous results / rejection / loading
    hideAllPanels();
  };
  reader.readAsDataURL(file);

  showToast(`Image loaded: ${file.name}`, 'success', 2500);
}

function formatFileSize(bytes) {
  if (bytes < 1024)        return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function clearImage() {
  selectedFile = null;

  const els = ['previewCard', 'loadingOverlay', 'resultsPanel', 'rejectionCard'];
  els.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.add('hidden');
  });

  const fileInput = document.getElementById('fileInput');
  if (fileInput) fileInput.value = '';

  const previewImage = document.getElementById('previewImage');
  if (previewImage) previewImage.src = '';

  window._lastPrediction = null;
  showToast('Image cleared.', 'info', 2000);
}

function hideAllPanels() {
  ['loadingOverlay', 'resultsPanel', 'rejectionCard'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.add('hidden');
  });
}

/* ── 6. Camera Capture ─────────────────────────────────────── */
let cameraStream = null;

function initCamera() {
  const openCameraBtn  = document.getElementById('openCameraBtn');
  const closeCameraBtn = document.getElementById('closeCameraBtn');
  const captureBtn     = document.getElementById('captureBtn');
  const cameraContainer= document.getElementById('cameraContainer');
  const cameraVideo    = document.getElementById('cameraVideo');

  if (!openCameraBtn) return;

  openCameraBtn.addEventListener('click', async () => {
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
      if (cameraVideo) {
        cameraVideo.srcObject = cameraStream;
        await cameraVideo.play();
      }
      if (cameraContainer) cameraContainer.classList.remove('hidden');
      openCameraBtn.classList.add('hidden');
    } catch (err) {
      showToast('Camera access denied or not available.', 'error');
      console.error('Camera error:', err);
    }
  });

  if (closeCameraBtn) {
    closeCameraBtn.addEventListener('click', stopCamera);
  }

  if (captureBtn) {
    captureBtn.addEventListener('click', () => {
      if (!cameraVideo) return;
      const canvas = document.createElement('canvas');
      canvas.width  = cameraVideo.videoWidth;
      canvas.height = cameraVideo.videoHeight;
      canvas.getContext('2d').drawImage(cameraVideo, 0, 0);

      canvas.toBlob((blob) => {
        const file = new File([blob], `camera_capture_${Date.now()}.jpg`, { type: 'image/jpeg' });
        handleFile(file);
        stopCamera();
      }, 'image/jpeg', 0.92);
    });
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  const cameraContainer = document.getElementById('cameraContainer');
  const openCameraBtn   = document.getElementById('openCameraBtn');
  const cameraVideo     = document.getElementById('cameraVideo');

  if (cameraContainer) cameraContainer.classList.add('hidden');
  if (openCameraBtn)   openCameraBtn.classList.remove('hidden');
  if (cameraVideo)     cameraVideo.srcObject = null;
}

/* ── 7. Sample Images ──────────────────────────────────────── */
function renderSampleImages() {
  const grid = document.getElementById('sampleGrid');
  if (!grid) return;

  grid.innerHTML = '';

  SAMPLE_IMAGES.forEach(sample => {
    const div = document.createElement('div');
    div.className = 'sample-thumb';
    div.title = sample.label;
    div.innerHTML = `
      <img src="${sample.url}" alt="${sample.label}" loading="lazy"
           onerror="this.style.display='none'; this.parentElement.querySelector('.sample-emoji').style.display='flex'">
      <div class="sample-emoji" style="display:none; align-items:center; justify-content:center; height:100%; font-size:1.5rem; background:#e8f5e9;">
        ${getCropEmojiFromName(sample.label)}
      </div>
      <div class="sample-label">${sample.label}</div>
    `;

    div.addEventListener('click', async () => {
      try {
        div.style.opacity = '0.6';
        const response = await fetch(sample.url);
        if (!response.ok) throw new Error('Sample image not found on server');
        const blob = await response.blob();
        const file = new File([blob], sample.name, { type: blob.type || 'image/jpeg' });
        handleFile(file);
      } catch (err) {
        // Fallback: use a placeholder blob
        showToast(`Loaded sample: ${sample.label}`, 'info', 2000);
        // Create a dummy file just to show the label — server will use sample path
        selectedFile = { _sampleUrl: sample.url, name: sample.name };
        hideAllPanels();
        const previewCard = document.getElementById('previewCard');
        const previewImage = document.getElementById('previewImage');
        const fileNameEl  = document.getElementById('previewFileName');
        const fileSizeEl  = document.getElementById('previewFileSize');
        if (previewImage)  previewImage.src = sample.url;
        if (fileNameEl)    fileNameEl.textContent = sample.name;
        if (fileSizeEl)    fileSizeEl.textContent = 'sample';
        if (previewCard)   previewCard.classList.remove('hidden');
      } finally {
        div.style.opacity = '1';
      }
    });

    grid.appendChild(div);
  });
}

function getCropEmojiFromName(name) {
  const lower = name.toLowerCase();
  for (const [key, emoji] of Object.entries(CROP_EMOJIS)) {
    if (lower.includes(key)) return emoji;
  }
  return CROP_EMOJIS.default;
}

/* ── 8. Prediction Flow ────────────────────────────────────── */
async function analyzeImage() {
  if (!selectedFile) {
    showToast('Please select or capture an image first.', 'warning');
    return;
  }

  const loadingOverlay = document.getElementById('loadingOverlay');
  const resultsPanel   = document.getElementById('resultsPanel');
  const rejectionCard  = document.getElementById('rejectionCard');

  // Show loading, hide others
  if (loadingOverlay) loadingOverlay.classList.remove('hidden');
  if (resultsPanel)   resultsPanel.classList.add('hidden');
  if (rejectionCard)  rejectionCard.classList.add('hidden');

  // Animate loading steps
  animateLoadingSteps();

  try {
    let formData = new FormData();

    // If it's a real File object
    if (selectedFile instanceof File) {
      formData.append('file', selectedFile);
    } else if (selectedFile._sampleUrl) {
      // For sample images that couldn't be fetched, send the URL
      formData.append('sample_url', selectedFile._sampleUrl);
      formData.append('filename',   selectedFile.name);
    }

    const response = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Server error ${response.status}: ${errText}`);
    }

    const data = await response.json();

    // Hide loading
    if (loadingOverlay) loadingOverlay.classList.add('hidden');

    if (data.status === 'rejected') {
      renderRejection(data);
    } else if (data.status === 'success' || data.prediction) {
      window._lastPrediction = data;
      renderResults(data);
    } else {
      throw new Error(data.message || 'Unexpected response from server.');
    }

  } catch (err) {
    console.error('Prediction error:', err);
    if (loadingOverlay) loadingOverlay.classList.add('hidden');
    showToast(`Analysis failed: ${err.message}`, 'error', 6000);

    // Show inline error
    renderError(err.message);
  }
}

function animateLoadingSteps() {
  const steps = document.querySelectorAll('.loading-step');
  steps.forEach(s => { s.classList.remove('active', 'done'); });

  let i = 0;
  const advance = () => {
    if (i > 0) steps[i - 1].classList.replace('active', 'done');
    if (i < steps.length) {
      steps[i].classList.add('active');
      i++;
      setTimeout(advance, 850);
    }
  };
  advance();
}

function renderRejection(data) {
  const card = document.getElementById('rejectionCard');
  if (!card) return;

  card.innerHTML = `
    <div class="rejection-card">
      <h3>🚫 Image Rejected</h3>
      <p><strong>Reason:</strong> ${escapeHtml(data.reason || data.message || 'Could not process this image.')}</p>
      <p style="margin-top:0.5rem; font-size:0.82rem; color:#888;">
        Please upload a clear, well-lit image of a plant leaf or crop. Avoid blurry, dark, or non-plant images.
      </p>
    </div>
  `;
  card.classList.remove('hidden');
}

function renderError(message) {
  const card = document.getElementById('rejectionCard');
  if (!card) return;

  card.innerHTML = `
    <div class="rejection-card">
      <h3>⚠️ Error</h3>
      <p>${escapeHtml(message)}</p>
    </div>
  `;
  card.classList.remove('hidden');
}

/* ── 9. Render Results ─────────────────────────────────────── */
function renderResults(data) {
  const panel = document.getElementById('resultsPanel');
  if (!panel) return;

  const pred        = data.prediction || data;
  const confidence  = Math.round((pred.confidence || 0) * 100);
  const isHealthy   = (pred.is_healthy === true) || (pred.disease_name || '').toLowerCase().includes('healthy');
  const cropName    = pred.crop_name    || pred.crop    || 'Unknown Crop';
  const diseaseName = pred.disease_name || pred.disease || 'Unknown';
  const topK        = pred.top_predictions || pred.alternatives || [];

  // Confidence level
  const confLevel = confidence >= 75 ? 'high' : confidence >= 50 ? 'medium' : 'low';
  const confColor = confLevel === 'high' ? '#1B5E20' : confLevel === 'medium' ? '#E65100' : '#B71C1C';

  // Header class
  const headerClass = isHealthy ? 'healthy' : confidence < 50 ? 'danger' : 'diseased';
  const emoji       = isHealthy ? '✅' : '🦠';

  // Disease info from prediction
  const info = pred.disease_info || data.disease_info || {};

  // Build tabs content
  function langField(field) {
    if (!field) return 'N/A';
    if (Array.isArray(field)) {
      return field.map(item => `• ${item}`).join('\n');
    }
    if (currentLang === 'te' && typeof field === 'object' && field.te) {
      if (Array.isArray(field.te)) return field.te.map(item => `• ${item}`).join('\n');
      return field.te;
    }
    if (typeof field === 'object' && field.en) {
      if (Array.isArray(field.en)) return field.en.map(item => `• ${item}`).join('\n');
      return field.en;
    }
    if (typeof field === 'string') return field;
    return 'N/A';
  }

  const symptoms   = langField(info.symptoms   || pred.symptoms);
  const causes     = langField(info.causes     || pred.causes);
  const prevention = langField(info.prevention || pred.prevention);
  const treatment  = langField(info.treatment  || pred.treatment);
  const expertNote = langField(info.expert_note|| pred.expert_note || info.management || info.extension_officer_note);

  // Advisory for low confidence
  const showAdvisory = confidence < 60;

  panel.innerHTML = `
    <!-- Header -->
    <div class="result-header ${headerClass}">
      <div class="result-emoji">${emoji}</div>
      <div class="result-meta">
        <div class="result-crop">${escapeHtml(cropName)}</div>
        <div class="result-disease">${escapeHtml(diseaseName)}</div>
        <span class="confidence-badge ${confLevel}">${confidence}% Confidence</span>
      </div>
    </div>

    <!-- Confidence bar -->
    <div class="confidence-bar-wrap">
      <div class="confidence-bar-label">
        <span>Model Confidence</span>
        <span style="color:${confColor}; font-weight:700;">${confidence}%</span>
      </div>
      <div class="confidence-bar-bg">
        <div class="confidence-bar-fill ${confLevel}" id="confBarFill" style="width:0%"></div>
      </div>
    </div>

    ${showAdvisory ? `
    <div class="advisory-box">
      ⚠️ <strong>Low-confidence prediction.</strong> Results may not be accurate. Please consult an agricultural expert or take a clearer photo in good lighting.
    </div>` : ''}

    <!-- Top-3 alternatives -->
    ${topK.length > 1 ? `
    <div class="alternatives">
      <h4>Top Predictions</h4>
      ${topK.slice(0, 3).map((alt, i) => {
        const pct = Math.round((alt.confidence || alt.probability || 0) * 100);
        return `
          <div class="alt-item">
            <div class="alt-name">${i === 0 ? '🥇' : i === 1 ? '🥈' : '🥉'} ${escapeHtml(alt.disease_name || alt.label || `Option ${i+1}`)}</div>
            <div class="alt-bar-bg"><div class="alt-bar-fill" data-pct="${pct}" style="width:0%"></div></div>
            <div class="alt-pct">${pct}%</div>
          </div>
        `;
      }).join('')}
    </div>` : ''}

    <!-- Tab Navigation -->
    <div class="result-tabs-nav">
      <button class="result-tab-btn active" data-tab="symptoms">🔍 Symptoms</button>
      <button class="result-tab-btn" data-tab="causes">🧬 Causes</button>
      <button class="result-tab-btn" data-tab="prevention">🛡️ Prevention</button>
      <button class="result-tab-btn" data-tab="treatment">💊 Treatment</button>
      <button class="result-tab-btn" data-tab="expert">👨‍🌾 Expert Note</button>
    </div>

    <div class="result-tab-content">
      <div class="result-tab-panel active" id="tab-symptoms">
        <div class="tab-section-title">🔍 Symptoms</div>
        <div class="tab-section-body">${formatTabContent(symptoms)}</div>
      </div>
      <div class="result-tab-panel" id="tab-causes">
        <div class="tab-section-title">🧬 Causes</div>
        <div class="tab-section-body">${formatTabContent(causes)}</div>
      </div>
      <div class="result-tab-panel" id="tab-prevention">
        <div class="tab-section-title">🛡️ Prevention Methods</div>
        <div class="tab-section-body">${formatTabContent(prevention)}</div>
      </div>
      <div class="result-tab-panel" id="tab-treatment">
        <div class="tab-section-title">💊 Treatment</div>
        <div class="tab-section-body">${formatTabContent(treatment)}</div>
      </div>
      <div class="result-tab-panel" id="tab-expert">
        <div class="tab-section-title">👨‍🌾 Expert / Extension Officer Note</div>
        <div class="officer-callout">
          <strong>📞 Agricultural Extension Officer Advisory</strong>
          ${expertNote !== 'N/A' ? escapeHtml(expertNote) : 'Contact your local Agricultural Extension Officer for personalized advice. They can assess your crop in person and recommend region-specific treatments.'}
        </div>
      </div>
    </div>

    <!-- Disclaimer -->
    <p class="result-disclaimer">
      ⚠️ This prediction is AI-generated and should be used as a guide only. Always verify with a certified agronomist before applying treatments.
    </p>

    <!-- Actions -->
    <div class="result-actions">
      <button class="btn btn-primary btn-sm" onclick="downloadReport(window._lastPrediction)">
        ⬇️ Download Report
      </button>
      <button class="btn btn-outline btn-sm" onclick="clearImage()">
        🔄 New Analysis
      </button>
    </div>
  `;

  panel.classList.remove('hidden');

  // Animate confidence bar
  requestAnimationFrame(() => {
    setTimeout(() => {
      const fill = document.getElementById('confBarFill');
      if (fill) fill.style.width = `${confidence}%`;

      // Animate alternative bars
      panel.querySelectorAll('.alt-bar-fill').forEach(bar => {
        bar.style.width = `${bar.dataset.pct}%`;
      });
    }, 120);
  });

  // Tab switching
  panel.querySelectorAll('.result-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      panel.querySelectorAll('.result-tab-btn').forEach(b => b.classList.remove('active'));
      panel.querySelectorAll('.result-tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const target = panel.querySelector(`#tab-${btn.dataset.tab}`);
      if (target) target.classList.add('active');
    });
  });
}

function formatTabContent(text) {
  if (!text || text === 'N/A') return '<em>Information not available.</em>';
  // Convert newlines to paragraphs, and bullet markers to list
  const lines = String(text).split('\n').filter(l => l.trim());
  const isList = lines.every(l => l.trim().startsWith('-') || l.trim().startsWith('•'));
  if (isList) {
    const items = lines.map(l => `<li>${escapeHtml(l.replace(/^[-•]\s*/, ''))}</li>`).join('');
    return `<ul>${items}</ul>`;
  }
  return lines.map(l => `<p>${escapeHtml(l)}</p>`).join('');
}

/* ── 10. Download Report ───────────────────────────────────── */
function downloadReport(data) {
  if (!data) {
    showToast('No prediction data to download.', 'warning');
    return;
  }

  const pred = data.prediction || data;
  const info = pred.disease_info || {};
  const confidence = Math.round((pred.confidence || 0) * 100);
  const timestamp = new Date().toLocaleString();

  const lines = [
    '═══════════════════════════════════════════════════',
    '           CROPSENSE AI — PREDICTION REPORT        ',
    '═══════════════════════════════════════════════════',
    '',
    `Generated : ${timestamp}`,
    `Language  : ${currentLang === 'te' ? 'Telugu' : 'English'}`,
    '',
    '─── DIAGNOSIS ──────────────────────────────────────',
    `Crop      : ${pred.crop_name    || pred.crop    || 'N/A'}`,
    `Disease   : ${pred.disease_name || pred.disease || 'N/A'}`,
    `Status    : ${pred.is_healthy ? '✅ HEALTHY' : '🦠 DISEASED'}`,
    `Confidence: ${confidence}%`,
    '',
    '─── DISEASE INFORMATION ─────────────────────────────',
    `Symptoms  : ${stringifyField(info.symptoms   || pred.symptoms)}`,
    '',
    `Causes    : ${stringifyField(info.causes     || pred.causes)}`,
    '',
    `Prevention: ${stringifyField(info.prevention || pred.prevention)}`,
    '',
    `Treatment : ${stringifyField(info.treatment  || pred.treatment)}`,
    '',
    '─── EXPERT NOTE ─────────────────────────────────────',
    stringifyField(info.expert_note || pred.expert_note || 'Contact your local Agricultural Extension Officer.'),
    '',
    '─── TOP PREDICTIONS ─────────────────────────────────',
    ...(pred.top_predictions || pred.alternatives || []).slice(0, 3).map((alt, i) => {
      const pct = Math.round((alt.confidence || alt.probability || 0) * 100);
      return `  ${i+1}. ${alt.disease_name || alt.label || 'N/A'} — ${pct}%`;
    }),
    '',
    '─── DISCLAIMER ──────────────────────────────────────',
    'This report is AI-generated and should be used as a guide only.',
    'Always verify results with a certified agronomist before applying',
    'any treatments or making farm management decisions.',
    '',
    '═══════════════════════════════════════════════════',
    '            CropSense AI — Powered by Deep Learning ',
    '═══════════════════════════════════════════════════',
  ];

  const text = lines.join('\n');
  const blob = new Blob([text], { type: 'text/plain' });
  const url  = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href     = url;
  a.download = `CropSense_Report_${Date.now()}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  showToast('Report downloaded successfully!', 'success');
}

function stringifyField(field) {
  if (!field) return 'N/A';
  if (Array.isArray(field)) return field.join('; ');
  if (typeof field === 'object' && (field.en || field.te)) return field[currentLang] || field.en || 'N/A';
  return String(field);
}

/* ── 11. Dashboard ─────────────────────────────────────────── */
async function loadDashboard() {
  await Promise.allSettled([loadStats(), loadHistory()]);
}

async function loadStats() {
  try {
    const resp = await fetch(`${API_BASE}/api/dashboard`);
    if (!resp.ok) throw new Error(`Stats fetch failed: ${resp.status}`);
    const json = await resp.json();
    const data = json.stats || json;
    renderStats(data);
    drawDiseaseChart(data.top_diseases || data.disease_distribution || []);
    drawHealthChart(data.healthy_count || 0, data.diseased_count || 0);
  } catch (err) {
    console.error('Stats error:', err);
    renderStatsError();
  }
}

function renderStats(data) {
  const grid = document.getElementById('statsGrid');
  if (!grid) return;

  const stats = [
    { label: 'Total Scans',     value: data.total_predictions ?? data.total_scans ?? data.total ?? 0, icon: '🔬', color: 'blue'   },
    { label: 'Healthy Crops',   value: data.healthy_count   ?? data.healthy   ?? 0, icon: '✅', color: 'green'  },
    { label: 'Diseased Crops',  value: data.diseased_count  ?? data.diseased  ?? 0, icon: '🦠', color: 'orange' },
    { label: 'Demo Predictions',value: data.demo_count      ?? 0,                   icon: '🧪', color: 'purple' },
  ];

  grid.innerHTML = stats.map(s => `
    <div class="stat-card">
      <div class="stat-icon ${s.color}">${s.icon}</div>
      <div class="stat-body">
        <div class="stat-label">${s.label}</div>
        <div class="stat-value">${s.value}</div>
      </div>
    </div>
  `).join('');
}

function renderStatsError() {
  const grid = document.getElementById('statsGrid');
  if (grid) grid.innerHTML = '<p class="text-muted" style="padding:1rem;">Could not load statistics.</p>';
}

async function loadHistory() {
  try {
    const resp = await fetch(`${API_BASE}/api/history?limit=20`);
    if (!resp.ok) throw new Error(`History fetch failed: ${resp.status}`);
    const data = await resp.json();
    renderHistory(Array.isArray(data) ? data : data.history || []);
  } catch (err) {
    console.error('History error:', err);
    const tbody = document.getElementById('historyTable');
    if (tbody) tbody.innerHTML = '<tr><td colspan="6" class="text-muted" style="padding:1rem;">Could not load history.</td></tr>';
  }
}

function renderHistory(rows) {
  const tbody = document.getElementById('historyTable');
  if (!tbody) return;

  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="6" style="padding:1.5rem; text-align:center; color:var(--text-muted);">No scan history yet. Analyze an image to get started!</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(row => {
    const conf     = Math.round((row.confidence || 0) * 100);
    const status   = row.is_healthy ? 'healthy' : row.status || 'diseased';
    const statusLabel = row.is_healthy ? 'Healthy' : row.status === 'rejected' ? 'Rejected' : 'Diseased';
    const date     = row.created_at ? new Date(row.created_at).toLocaleDateString() : 'N/A';

    return `
      <tr>
        <td>${date}</td>
        <td>${escapeHtml(row.crop_name || row.crop || 'N/A')}</td>
        <td>${escapeHtml(row.disease_name || row.disease || 'N/A')}</td>
        <td><span class="status-pill ${status}">${statusLabel}</span></td>
        <td>${conf}%</td>
        <td>
          <button class="btn btn-danger btn-sm" onclick="deleteHistoryRow('${row.id}')">🗑️</button>
        </td>
      </tr>
    `;
  }).join('');
}

async function deleteHistoryRow(id) {
  if (!confirm('Delete this scan record?')) return;
  try {
    const resp = await fetch(`${API_BASE}/api/history/${id}`, { method: 'DELETE' });
    if (!resp.ok) throw new Error('Delete failed');
    showToast('Record deleted.', 'success', 2000);
    await loadDashboard();
  } catch (err) {
    showToast('Could not delete record.', 'error');
  }
}

async function clearAllHistory() {
  if (!confirm('Clear ALL scan history? This action cannot be undone.')) return;
  try {
    const resp = await fetch(`${API_BASE}/api/history`, { method: 'DELETE' });
    if (!resp.ok) throw new Error('Clear failed');
    showToast('All history cleared.', 'success');
    await loadDashboard();
  } catch (err) {
    showToast('Could not clear history.', 'error');
  }
}

/* ── 12. Canvas Charts ─────────────────────────────────────── */
function drawDiseaseChart(distribution) {
  const canvas = document.getElementById('diseaseChart');
  if (!canvas) return;

  // Handle empty data
  const data = Array.isArray(distribution) && distribution.length
    ? distribution
    : [{ label: 'No Data', count: 1 }];

  const ctx    = canvas.getContext('2d');
  const W      = canvas.offsetWidth || 400;
  const H      = 220;

  canvas.width  = W;
  canvas.height = H;

  const palette = ['#2E7D32','#66BB6A','#F57F17','#C62828','#1565C0','#7B1FA2','#00695C','#F9A825'];
  const maxCount= Math.max(...data.map(d => d.count || d.value || 0), 1);
  const padL = 50, padR = 20, padT = 20, padB = 50;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;
  const barW   = Math.max(20, (chartW / data.length) * 0.65);
  const gapW   = chartW / data.length;

  ctx.clearRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = '#E8F0E9';
  ctx.lineWidth   = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + chartH - (i / 4) * chartH;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = '#888';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(Math.round((i / 4) * maxCount), padL - 6, y + 4);
  }

  // Bars
  data.forEach((item, i) => {
    const count  = item.count || item.value || 0;
    const barH   = (count / maxCount) * chartH;
    const x      = padL + i * gapW + (gapW - barW) / 2;
    const y      = padT + chartH - barH;

    // Bar gradient
    const grad = ctx.createLinearGradient(0, y, 0, padT + chartH);
    grad.addColorStop(0, palette[i % palette.length]);
    grad.addColorStop(1, palette[i % palette.length] + '88');
    ctx.fillStyle = grad;

    // Rounded top
    const r = Math.min(5, barW / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + barW - r, y);
    ctx.quadraticCurveTo(x + barW, y, x + barW, y + r);
    ctx.lineTo(x + barW, padT + chartH);
    ctx.lineTo(x, padT + chartH);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
    ctx.fill();

    // Count label on bar
    ctx.fillStyle = '#333';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    if (barH > 20) ctx.fillText(count, x + barW / 2, y - 5);

    // X label
    ctx.fillStyle = '#555';
    ctx.font = '10px Inter, sans-serif';
    const lbl = (item.label || item.disease || item.name || `D${i+1}`).substring(0, 12);
    ctx.save();
    ctx.translate(x + barW / 2, padT + chartH + 10);
    ctx.rotate(-Math.PI / 5);
    ctx.textAlign = 'right';
    ctx.fillText(lbl, 0, 0);
    ctx.restore();
  });

  // Axis
  ctx.strokeStyle = '#ccc';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + chartH); ctx.lineTo(W - padR, padT + chartH); ctx.stroke();
}

function drawHealthChart(healthy, diseased) {
  const canvas = document.getElementById('healthChart');
  if (!canvas) return;

  const ctx    = canvas.getContext('2d');
  const size   = Math.min(canvas.offsetWidth || 220, 220);

  canvas.width  = size;
  canvas.height = size;

  const total = healthy + diseased || 1;
  const cx    = size / 2;
  const cy    = size / 2;
  const r     = size * 0.36;
  const inner = size * 0.22;

  const segments = [
    { value: healthy,  color: '#4CAF50', label: 'Healthy' },
    { value: diseased, color: '#F57F17', label: 'Diseased' },
  ];

  let startAngle = -Math.PI / 2;

  ctx.clearRect(0, 0, size, size);

  // Draw segments
  segments.forEach(seg => {
    const slice = (seg.value / total) * 2 * Math.PI;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, startAngle, startAngle + slice);
    ctx.closePath();
    ctx.fillStyle = seg.color;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 3;
    ctx.stroke();
    startAngle += slice;
  });

  // Donut hole
  ctx.beginPath();
  ctx.arc(cx, cy, inner, 0, 2 * Math.PI);
  ctx.fillStyle = '#fff';
  ctx.fill();

  // Center text
  ctx.fillStyle = '#1A2E1B';
  ctx.font = `bold ${Math.round(size * 0.1)}px Inter, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(`${Math.round((healthy / total) * 100)}%`, cx, cy - 6);
  ctx.font = `${Math.round(size * 0.065)}px Inter, sans-serif`;
  ctx.fillStyle = '#6B7C6E';
  ctx.fillText('Healthy', cx, cy + size * 0.065);

  // Legend
  const legendY = size - 28;
  segments.forEach((seg, i) => {
    const lx = i === 0 ? cx - 55 : cx + 5;
    ctx.fillStyle = seg.color;
    ctx.fillRect(lx, legendY, 12, 12);
    ctx.fillStyle = '#555';
    ctx.font = `${Math.round(size * 0.07)}px Inter, sans-serif`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`${seg.label} (${seg.value})`, lx + 16, legendY);
  });
}

/* ── 13. Disease Browser ───────────────────────────────────── */
let allDiseases    = [];
let activeFilter   = 'All';
let searchQuery    = '';

async function loadDiseases() {
  try {
    const resp = await fetch(`${API_BASE}/api/diseases`);
    if (!resp.ok) throw new Error(`Diseases fetch failed: ${resp.status}`);
    const data = await resp.json();
    allDiseases = Array.isArray(data) ? data : data.diseases || [];
    renderFilterChips();
    renderDiseaseCards();
  } catch (err) {
    console.error('Diseases error:', err);
    const grid = document.getElementById('diseasesGrid');
    if (grid) grid.innerHTML = '<p class="text-muted" style="grid-column:1/-1; padding:2rem; text-align:center;">Could not load disease database. Make sure the backend is running.</p>';
  }
}

function renderFilterChips() {
  const wrap = document.getElementById('filterChips');
  if (!wrap) return;

  const crops = ['All', ...new Set(allDiseases.map(d => d.crop_name || d.crop || 'Other').filter(Boolean))];

  wrap.innerHTML = crops.map(crop => `
    <button class="filter-chip ${crop === activeFilter ? 'active' : ''}" data-crop="${crop}">${crop}</button>
  `).join('');

  wrap.querySelectorAll('.filter-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      activeFilter = chip.dataset.crop;
      wrap.querySelectorAll('.filter-chip').forEach(c => c.classList.toggle('active', c.dataset.crop === activeFilter));
      renderDiseaseCards();
    });
  });
}

function renderDiseaseCards() {
  const grid = document.getElementById('diseasesGrid');
  if (!grid) return;

  let filtered = allDiseases;

  if (activeFilter !== 'All') {
    filtered = filtered.filter(d => (d.crop_name || d.crop || '') === activeFilter);
  }

  if (searchQuery) {
    const q = searchQuery.toLowerCase();
    filtered = filtered.filter(d =>
      (d.disease_name || d.name || '').toLowerCase().includes(q) ||
      (d.crop_name    || d.crop || '').toLowerCase().includes(q) ||
      (d.pathogen_type|| '').toLowerCase().includes(q)
    );
  }

  if (!filtered.length) {
    grid.innerHTML = `
      <div class="empty-state" style="grid-column:1/-1;">
        <div class="empty-icon">🔍</div>
        <p>No diseases found matching your search.</p>
      </div>
    `;
    return;
  }

  grid.innerHTML = filtered.map((disease, idx) => {
    const crop     = disease.crop_name    || disease.crop    || 'Unknown';
    const name     = disease.disease_name || disease.name    || 'Unknown';
    const pathogen = disease.pathogen_type|| disease.type    || (disease.is_healthy ? 'healthy' : 'other');
    const emoji    = getCropEmojiFromName(crop);
    const badgeClass = pathogenBadgeClass(pathogen, disease.is_healthy);

    return `
      <div class="disease-card" data-idx="${idx}" onclick="openDiseaseModal(${idx})">
        <div class="disease-card-emoji">${emoji}</div>
        <h3>${escapeHtml(name)}</h3>
        <div class="crop-name">${escapeHtml(crop)}</div>
        <span class="pathogen-badge ${badgeClass}">${capitalize(pathogen)}</span>
      </div>
    `;
  }).join('');

  // Store filtered for modal access
  window._filteredDiseases = filtered;
}

function pathogenBadgeClass(pathogen, isHealthy) {
  if (isHealthy || (pathogen || '').toLowerCase() === 'healthy') return 'healthy';
  const p = (pathogen || '').toLowerCase();
  if (p.includes('fung')) return 'fungal';
  if (p.includes('bact')) return 'bacterial';
  if (p.includes('vir'))  return 'viral';
  return 'other';
}

function initDiseaseSearch() {
  const input = document.getElementById('diseaseSearch');
  if (!input) return;
  input.addEventListener('input', () => {
    searchQuery = input.value.trim();
    renderDiseaseCards();
  });
}

/* ── 14. Disease Modal ─────────────────────────────────────── */
function openDiseaseModal(idx) {
  const diseases = window._filteredDiseases || allDiseases;
  const disease  = diseases[idx];
  if (!disease) return;

  const modal    = document.getElementById('diseaseModal');
  const titleEl  = document.getElementById('modalTitle');
  const bodyEl   = document.getElementById('modalBody');

  if (!modal) return;

  if (titleEl) titleEl.textContent = disease.disease_name || disease.name || 'Disease Detail';

  const info = disease.disease_info || disease;

  function f(field) {
    if (!field) return 'N/A';
    if (typeof field === 'object') return field[currentLang] || field.en || JSON.stringify(field);
    return String(field);
  }

  bodyEl.innerHTML = `
    <div class="modal-tabs-nav">
      <button class="modal-tab-btn active" data-tab="overview">Overview</button>
      <button class="modal-tab-btn" data-tab="symptoms">Symptoms</button>
      <button class="modal-tab-btn" data-tab="causes">Causes</button>
      <button class="modal-tab-btn" data-tab="prevention">Prevention</button>
      <button class="modal-tab-btn" data-tab="treatment">Treatment</button>
    </div>

    <div class="modal-tab-panel active" id="modal-tab-overview">
      <div class="modal-field"><div class="modal-field-label">Crop</div><div class="modal-field-value">${escapeHtml(disease.crop_name || disease.crop || 'N/A')}</div></div>
      <div class="modal-field"><div class="modal-field-label">Disease Name</div><div class="modal-field-value">${escapeHtml(disease.disease_name || disease.name || 'N/A')}</div></div>
      <div class="modal-field"><div class="modal-field-label">Pathogen Type</div><div class="modal-field-value"><span class="pathogen-badge ${pathogenBadgeClass(disease.pathogen_type, disease.is_healthy)}">${capitalize(disease.pathogen_type || 'N/A')}</span></div></div>
      <div class="modal-field"><div class="modal-field-label">Scientific Name</div><div class="modal-field-value"><em>${escapeHtml(disease.scientific_name || info.scientific_name || 'N/A')}</em></div></div>
      <div class="modal-field"><div class="modal-field-label">Status</div><div class="modal-field-value"><span class="status-pill ${disease.is_healthy ? 'healthy' : 'diseased'}">${disease.is_healthy ? 'Healthy' : 'Disease'}</span></div></div>
    </div>

    <div class="modal-tab-panel" id="modal-tab-symptoms">
      <div class="modal-field-label">Symptoms</div>
      <div class="modal-field-value">${formatTabContent(f(info.symptoms || disease.symptoms))}</div>
    </div>

    <div class="modal-tab-panel" id="modal-tab-causes">
      <div class="modal-field-label">Causes</div>
      <div class="modal-field-value">${formatTabContent(f(info.causes || disease.causes))}</div>
    </div>

    <div class="modal-tab-panel" id="modal-tab-prevention">
      <div class="modal-field-label">Prevention</div>
      <div class="modal-field-value">${formatTabContent(f(info.prevention || disease.prevention))}</div>
    </div>

    <div class="modal-tab-panel" id="modal-tab-treatment">
      <div class="modal-field-label">Treatment</div>
      <div class="modal-field-value">${formatTabContent(f(info.treatment || disease.treatment))}</div>
      ${info.expert_note || disease.expert_note ? `
      <div class="officer-callout" style="margin-top:1rem;">
        <strong>👨‍🌾 Expert Note</strong>
        ${escapeHtml(f(info.expert_note || disease.expert_note))}
      </div>` : ''}
    </div>
  `;

  // Tab switching in modal
  bodyEl.querySelectorAll('.modal-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      bodyEl.querySelectorAll('.modal-tab-btn').forEach(b => b.classList.remove('active'));
      bodyEl.querySelectorAll('.modal-tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const panel = bodyEl.querySelector(`#modal-tab-${btn.dataset.tab}`);
      if (panel) panel.classList.add('active');
    });
  });

  modal.classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeDiseaseModal() {
  const modal = document.getElementById('diseaseModal');
  if (modal) {
    modal.classList.remove('open');
    document.body.style.overflow = '';
  }
}

function initModal() {
  const modal   = document.getElementById('diseaseModal');
  const closeBtn= document.getElementById('modalCloseBtn');

  if (!modal) return;

  // Close on overlay click
  modal.addEventListener('click', (e) => {
    if (e.target === modal) closeDiseaseModal();
  });

  if (closeBtn) closeBtn.addEventListener('click', closeDiseaseModal);

  // ESC key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.classList.contains('open')) closeDiseaseModal();
  });
}

/* ── 15. Scroll to Section ─────────────────────────────────── */
function scrollToSection(id) {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: 'smooth' });
}

/* ── 16. Utility Functions ─────────────────────────────────── */
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g,  '&amp;')
    .replace(/</g,  '&lt;')
    .replace(/>/g,  '&gt;')
    .replace(/"/g,  '&quot;')
    .replace(/'/g,  '&#039;');
}

function capitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

/* ── 17. Smooth Hero CTA ───────────────────────────────────── */
function initHeroCTA() {
  const ctaBtn = document.getElementById('heroCTA');
  if (ctaBtn) {
    ctaBtn.addEventListener('click', () => scrollToSection('detect'));
  }

  const learnMoreBtn = document.getElementById('heroLearnMore');
  if (learnMoreBtn) {
    learnMoreBtn.addEventListener('click', () => scrollToSection('how-it-works'));
  }
}

/* ── 18. Init Dashboard Controls ──────────────────────────── */
function initDashboardControls() {
  const refreshBtn  = document.getElementById('refreshDashboard');
  const clearAllBtn = document.getElementById('clearAllHistory');

  if (refreshBtn)  refreshBtn.addEventListener('click',  loadDashboard);
  if (clearAllBtn) clearAllBtn.addEventListener('click', clearAllHistory);
}

/* ── 19. Intersection-based Section Loading ────────────────── */
function initLazySections() {
  let dashboardLoaded = false;
  let diseasesLoaded  = false;

  const dashboardSection = document.getElementById('dashboard');
  const diseasesSection  = document.getElementById('diseases');

  const sectionObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      if (entry.target === dashboardSection && !dashboardLoaded) {
        dashboardLoaded = true;
        loadDashboard();
      }
      if (entry.target === diseasesSection && !diseasesLoaded) {
        diseasesLoaded = true;
        loadDiseases();
      }
    });
  }, { rootMargin: '0px 0px -10% 0px' });

  if (dashboardSection) sectionObserver.observe(dashboardSection);
  if (diseasesSection)  sectionObserver.observe(diseasesSection);
}

/* ── 20. App Init (DOMContentLoaded) ──────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  // Core modules
  initNavigation();
  initUpload();
  initCamera();
  renderSampleImages();
  initHeroCTA();
  initDashboardControls();
  initDiseaseSearch();
  initModal();

  // Lazy-load dashboard + diseases when sections come into view
  initLazySections();

  // Apply default language
  setLanguage(currentLang);

  // Create toast container if not present in HTML
  if (!document.getElementById('toastContainer')) {
    const tc = document.createElement('div');
    tc.id = 'toastContainer';
    document.body.appendChild(tc);
  }

  console.log('%c🌿 CropSense AI loaded.', 'color:#2E7D32; font-weight:bold; font-size:14px;');
});

/* ── 21. Expose global helpers ─────────────────────────────── */
window.analyzeImage       = analyzeImage;
window.clearImage         = clearImage;
window.downloadReport     = downloadReport;
window.setLanguage        = setLanguage;
window.scrollToSection    = scrollToSection;
window.openDiseaseModal   = openDiseaseModal;
window.closeDiseaseModal  = closeDiseaseModal;
window.deleteHistoryRow   = deleteHistoryRow;
window.clearAllHistory    = clearAllHistory;
window.loadDashboard      = loadDashboard;
window.stopCamera         = stopCamera;
