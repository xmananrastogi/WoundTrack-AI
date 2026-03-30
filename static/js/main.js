// ── BOOTSTRAP JINJA VARS ─────────────────────────────────────────────────────
function loadJsonData(id, fallback) {
  var el = document.getElementById(id);
  if (!el) return fallback;
  var text = el.textContent.trim();
  if (!text || text.startsWith('{' + '{')) return fallback;
  try { return JSON.parse(text); } catch(e) { return fallback; }
}

const METRIC_INFO  = loadJsonData('metric_info_data', {});
const SCAFFOLD_DB  = loadJsonData('scaffold_db_data', {});
const CORR_JSON    = loadJsonData('corr_json_data', {"data":[]});
const BOX_JSON     = loadJsonData('box_json_data', {"data":[]});
const STIFF_JSON   = loadJsonData('stiff_json_data', {"data":[]});

const METRICS_COMPARE = [
  'final_closure_pct','healing_rate_um2_per_hr','r_squared',
  'sigmoid_max_rate_pct_hr','sigmoid_lag_phase_hr','time_to_50_closure_hr',
  'edge_asymmetry_index','migration_fraction','mean_velocity_um_min',
  'migration_efficiency_mean','num_cells_tracked','msd_alpha'
];

let allExperimentsData = [];
let currentModalId = null;
let currentPubId   = null;

// ── PLOTLY THEME ─────────────────────────────────────────────────────────────
function pLayout(base) {
  base = base || {};
  return Object.assign({
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#94a3b8', family: 'Inter, sans-serif', size: 11 },
    xaxis: {
      gridcolor: 'rgba(255,255,255,0.05)',
      zerolinecolor: 'rgba(255,255,255,0.1)',
      tickfont: { size: 10 }
    },
    yaxis: {
      gridcolor: 'rgba(255,255,255,0.05)',
      zerolinecolor: 'rgba(255,255,255,0.1)',
      tickfont: { size: 10 }
    },
    colorway: ['#2dd4bf', '#22d3ee', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444'],
    margin: { t: 40, r: 20, l: 50, b: 50 },
    hovermode: 'closest',
    dragmode: 'pan'
  }, base);
}

window.switchTab = function(e, name) {
  document.querySelectorAll('.tab-pane').forEach(function(t){ t.classList.remove('active'); });
  document.querySelectorAll('.s-tab').forEach(function(b){ b.classList.remove('active'); });
  document.getElementById(name).classList.add('active');
  var btn = e.target ? e.target.closest('.s-tab') : e;
  if (btn) btn.classList.add('active');
  if (name==='stats') renderStatsPlots();
  if (name==='compare' && !allExperimentsData.length) loadCompareData();
  if (name==='scaffold') initScaffoldDesigner();
};

// ── RENDER STATS PLOTS ───────────────────────────────────────────────────────
function renderStatsPlots() {
  var noData = '<p style="color:var(--text-mute);font-family:var(--mono);text-align:center;padding:50px;font-size:.8rem">Insufficient data</p>';
  try {
    if (STIFF_JSON && STIFF_JSON.data && STIFF_JSON.data.length) Plotly.newPlot('stiffness-scatter', STIFF_JSON.data, pLayout(STIFF_JSON.layout), {responsive:true});
    else document.getElementById('stiffness-scatter').innerHTML = noData;
  } catch(e) { document.getElementById('stiffness-scatter').innerHTML = noData; }
  try {
    if (BOX_JSON && BOX_JSON.data) Plotly.newPlot('box-plots', BOX_JSON.data, pLayout(BOX_JSON.layout), {responsive:true});
    else document.getElementById('box-plots').innerHTML = noData;
  } catch(e) { document.getElementById('box-plots').innerHTML = noData; }
  try {
    if (CORR_JSON && CORR_JSON.data) Plotly.newPlot('corr-heatmap', CORR_JSON.data, pLayout(CORR_JSON.layout), {responsive:true});
    else document.getElementById('corr-heatmap').innerHTML = noData;
  } catch(e) { document.getElementById('corr-heatmap').innerHTML = noData; }
}

// ── FILTER & OVERLAYS ────────────────────────────────────────────────────────
window.filterCards = function(e, cond) {
  document.querySelectorAll('.exp-card').forEach(function(c){
    c.style.display = (cond === 'all' || c.dataset.condition === cond) ? '' : 'none';
  });
  document.querySelectorAll('.filter-pill').forEach(function(b){ b.classList.remove('active'); });
  e.target.classList.add('active');
};

window.setOverlay = function(btn, mode, expId) {
  btn.closest('.overlay-controls').querySelectorAll('.ov-btn').forEach(function(b){ b.classList.remove('active'); });
  btn.classList.add('active');
  var img   = document.getElementById('img-raw-' + expId);
  var label = document.getElementById('overlay-label-' + expId);
  if (!img) return;
  var MODES = {
    raw:     { filter:'none', label:'Raw · Analysis' },
    heatmap: { filter:'hue-rotate(200deg) saturate(3.5) contrast(1.4)', label:'Heatmap · False colour' },
    mask:    { filter:'grayscale(1) contrast(5) brightness(.7)', label:'Mask · Segmentation' }
  };
  var m = MODES[mode] || MODES.raw;
  img.style.filter = m.filter;
  if (label) label.textContent = m.label;
};

// ── CANVAS PLAYERS ────────────────────────────────────────────────────────────
var playerFrames=[], playerIdx=0, playerPlaying=false, playerTimer=null, playerFPS=8;

window.openPlayer = function(expId) {
  document.getElementById('playerModalTitle').textContent = 'Timelapse — ' + expId;
  document.getElementById('playerModal').classList.add('active');
  loadPlayerFrames(expId, 'standalone');
};

async function updateLabInsights(data) {
  var box = document.getElementById('labInsightBox');
  var text = document.getElementById('labInsightText');
  if (!data) { box.style.display = 'none'; return; }

  var insights = [];
  var rate = data.sigmoid_max_rate_pct_hr || data.healing_rate_um2_per_hr || 0;
  var persistence = data.migration_efficiency_mean || data.meander_index_mean || 0;
  var drug = data.treatment || 'the applied treatment';

  // 1. Healing Rate Insight
  if (rate > 10) {
    insights.push("Significant proliferative vigor observed; the cell sheet is closing the gap at an accelerated rate.");
  } else if (rate < 2 && data.num_timepoints > 10) {
    insights.push("The healing kinetics appear suppressed, potentially due to contact inhibition or cytotoxic substrate interaction.");
  }

  // 2. Migration Insight
  if (persistence > 0.7) {
    insights.push("High directional persistence detected (Meander Index: " + persistence.toFixed(2) + "), suggesting a highly polarized chemotactic response to the wound edge.");
  } else if (persistence < 0.3 && persistence > 0) {
    insights.push("Migration patterns exhibit high stochasticity (random walk behavior), suggesting a lack of directional cues or scaffold-induced wandering.");
  }

  // 3. Treatment specific
  if (data.substrate_material === 'Collagen I') {
    insights.push("The Collagen I matrix is effectively promoting integrin-mediated adhesion, facilitating cohesive sheet migration.");
  }

  if (insights.length > 0) {
    box.style.display = 'block';
    text.innerHTML = "📝 " + insights.join(" ");
  } else {
    box.style.display = 'none';
  }
}

async function loadPlayerFrames(expId, mode) {
  mode = mode || 'modal';
  try {
    var res = await fetch('/api/gallery_frames/' + expId);
    var d   = await res.json();
    var frames = d.frames || [];
    if (mode === 'standalone') {
      saFrames = frames; saIdx = 0; saPlaying = false; initSAPlayer(frames);
    } else {
      playerFrames = frames; playerIdx = 0; playerPlaying = false; initModalPlayer(frames);
    }
  } catch(e) { console.error('Frame load error', e); }
}

function initModalPlayer(frames) {
  var wrap  = document.getElementById('modalPlayerWrap');
  var noMsg = document.getElementById('noPlayerMsg');
  if (!frames.length) { wrap.style.display='none'; noMsg.style.display='block'; return; }
  wrap.style.display='block'; noMsg.style.display='none';
  var scrubber = document.getElementById('playerScrubber');
  scrubber.max = frames.length - 1; scrubber.value = 0;
  document.getElementById('frameCounter').textContent = '1/' + frames.length;
  drawPlayerFrame(document.getElementById('modalCanvas'), frames, 0);
}

window.togglePlay = function() {
  playerPlaying = !playerPlaying;
  document.getElementById('playPauseBtn').textContent = playerPlaying ? '⏸' : '▶';
  if (playerPlaying) runPlayer(); else clearTimeout(playerTimer);
};

function runPlayer() {
  if (!playerPlaying || !playerFrames.length) return;
  playerIdx = (playerIdx + 1) % playerFrames.length;
  document.getElementById('playerScrubber').value = playerIdx;
  document.getElementById('frameCounter').textContent = (playerIdx+1) + '/' + playerFrames.length;
  drawPlayerFrame(document.getElementById('modalCanvas'), playerFrames, playerIdx);
  playerTimer = setTimeout(runPlayer, 1000 / playerFPS);
}

window.scrubTo = function(v) {
  playerIdx = parseInt(v);
  document.getElementById('frameCounter').textContent = (playerIdx+1) + '/' + playerFrames.length;
  drawPlayerFrame(document.getElementById('modalCanvas'), playerFrames, playerIdx);
};
window.setFPS = function(v) { playerFPS = parseInt(v); };

function drawPlayerFrame(canvas, frames, idx) {
  if (!frames[idx]) return;
  var img = new Image();
  img.onload = function() {
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    var ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    ctx.fillStyle = 'rgba(10, 12, 16, 0.8)';
    ctx.roundRect(12, 12, 140, 26, 6);
    ctx.fill();
    ctx.fillStyle = '#2dd4bf';
    ctx.font = '600 11px Inter, sans-serif';
    ctx.fillText('FRAME ' + (idx + 1) + ' / ' + frames.length, 22, 29);
  };
  img.src = frames[idx];
}

// Standalone player
var saFrames=[], saIdx=0, saPlaying=false, saTimer=null, saFPS=8;
function initSAPlayer(frames) {
  var canvas = document.getElementById('standaloneCanvas');
  document.getElementById('saScrubber').max = frames.length - 1;
  document.getElementById('saScrubber').value = 0;
  document.getElementById('saFrameCounter').textContent = '1/' + frames.length;
  document.getElementById('saInfo').textContent = frames.length ? (frames.length + ' frames loaded') : 'No frames available';
  if (frames.length) drawPlayerFrame(canvas, frames, 0);
}
window.saTogglePlay = function() {
  saPlaying = !saPlaying;
  document.getElementById('saPlayPauseBtn').textContent = saPlaying ? '⏸' : '▶';
  if (saPlaying) runSAPlayer(); else clearTimeout(saTimer);
};
function runSAPlayer() {
  if (!saPlaying || !saFrames.length) return;
  saIdx = (saIdx + 1) % saFrames.length;
  document.getElementById('saScrubber').value = saIdx;
  document.getElementById('saFrameCounter').textContent = (saIdx+1) + '/' + saFrames.length;
  drawPlayerFrame(document.getElementById('standaloneCanvas'), saFrames, saIdx);
  saTimer = setTimeout(runSAPlayer, 1000 / saFPS);
}
window.saScrubTo = function(v) {
  saIdx = parseInt(v);
  document.getElementById('saFrameCounter').textContent = (saIdx+1) + '/' + saFrames.length;
  drawPlayerFrame(document.getElementById('standaloneCanvas'), saFrames, saIdx);
};
window.saSetFPS = function(v) { saFPS = parseInt(v); };
window.closePlayerModal = function() {
  document.getElementById('playerModal').classList.remove('active');
  saPlaying = false; clearTimeout(saTimer);
};

// ── FILE UPLOAD & PIPELINE ───────────────────────────────────────────────────
window.onFileSelected = function() {
  var f = document.getElementById('fileUpload').files[0];
  document.getElementById('fileName').textContent = f ? ('Selected: ' + f.name) : '';
};

var dz = document.getElementById('dropZone');
['dragenter','dragover','dragleave','drop'].forEach(function(e){ dz.addEventListener(e, function(ev){ ev.preventDefault(); }); });
['dragenter','dragover'].forEach(function(e){ dz.addEventListener(e, function(){ dz.style.borderColor='var(--bio-green)'; }); });
['dragleave','drop'].forEach(function(e){ dz.addEventListener(e, function(){ dz.style.borderColor=''; }); });
dz.addEventListener('drop', function(ev) {
  document.getElementById('fileUpload').files = ev.dataTransfer.files;
  onFileSelected();
});
dz.addEventListener('click', function(){ document.getElementById('fileUpload').click(); });

window.startUpload = function() {
  var file = document.getElementById('fileUpload').files[0];
  if (!file) { updateAcq(0,'No file selected','error'); return; }
  document.getElementById('analyzeBtn').disabled = true;
  showAcq();
  updateAcq(5,'Uploading data…');
  var fd = new FormData();
  fd.append('file', file);
  fd.append('frameInterval', document.getElementById('frameInterval').value);
  fetch('/api/upload', {method:'POST', body:fd})
    .then(function(r){ return r.json(); })
    .then(function(d) {
      if (!d.analysis_id) throw new Error(d.error || 'Upload failed');
      updateAcq(15,'File received — starting pipeline…');
      startAnalysis(d.analysis_id);
    })
    .catch(function(e) {
      updateAcq(0, 'Upload failed: ' + e.message, 'error');
      document.getElementById('analyzeBtn').disabled = false;
    });
};

function startAnalysis(id) {
  var body = {
    analysis_id:             id,
    disk_size:               document.getElementById('diskSize').value,
    time_interval:           document.getElementById('timeInterval').value,
    pixel_scale:             document.getElementById('pixelScale').value,
    sample_id:               document.getElementById('sampleId').value,
    substrate_material:      document.getElementById('substrateMaterial').value,
    substrate_stiffness_kpa: document.getElementById('substrateStiffness').value || null,
    treatment:               document.getElementById('treatment').value
  };
  fetch('/api/analyze', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)})
    .then(function(r){ return r.json(); })
    .then(function(d) {
      if (d.status === 'started') pollProgress(d.analysis_id);
      else throw new Error('Failed to start');
    })
    .catch(function(e) {
      updateAcq(0, 'Start failed: ' + e.message, 'error');
      document.getElementById('analyzeBtn').disabled = false;
    });
}

function pollProgress(id) {
  fetch('/api/status?analysis_id=' + id)
    .then(function(r){ return r.json(); })
    .then(function(d) {
      updateAcq(d.progress || 0, d.status || '');
      document.getElementById('acqPct').textContent = (d.progress || 0) + '%';
      drawWaveform(d.progress || 0);
      if (d.running) {
        setTimeout(function(){ pollProgress(id); }, 900);
      } else if (d.progress >= 100) {
        updateAcq(100, 'Analysis complete — reloading…');
        setTimeout(function(){ location.reload(); }, 1800);
      } else if ((d.status || '').indexOf('❌') !== -1) {
        updateAcq(0, d.status, 'error');
        document.getElementById('analyzeBtn').disabled = false;
      }
    })
    .catch(function(){ setTimeout(function(){ pollProgress(id); }, 1500); });
}

// ── ACQUISITION PANEL ────────────────────────────────────────────────────────
function showAcq() {
  document.getElementById('acqPanel').classList.add('active');
  updateAcqTimestamp();
  initWaveCanvas();
}
function updateAcq(pct, status, type) {
  type = type || 'info';
  document.getElementById('acqFill').style.width = pct + '%';
  document.getElementById('acqStatus').textContent = status;
  document.getElementById('acqStatus').style.color = type === 'error' ? 'var(--bio-red)' : 'var(--bio-green)';
}
function updateAcqTimestamp() {
  var el = document.getElementById('acqTimestamp');
  if (!el || !document.getElementById('acqPanel').classList.contains('active')) return;
  el.textContent = new Date().toTimeString().slice(0,8);
  setTimeout(updateAcqTimestamp, 1000);
}

var waveCtx = null, wavePoints = [];
function initWaveCanvas() {
  var canvas = document.getElementById('waveCanvas');
  if (!canvas) return;
  canvas.width  = canvas.offsetWidth || 800;
  canvas.height = 120;
  waveCtx = canvas.getContext('2d');
  wavePoints = Array(canvas.width).fill(60);
}
function drawWaveform(progress) {
  if (!waveCtx) return;
  var w = waveCtx.canvas.width, h = waveCtx.canvas.height;
  waveCtx.clearRect(0, 0, w, h);
  wavePoints.shift();
  wavePoints.push((h/2) - ((progress/100)*h*0.4) + (Math.random()-0.5)*8);
  waveCtx.beginPath();
  waveCtx.strokeStyle = '#2ecc71';
  waveCtx.lineWidth = 1.5;
  waveCtx.shadowBlur = 8;
  waveCtx.shadowColor = 'rgba(46,204,113,0.5)';
  wavePoints.forEach(function(y, i){ if (i===0) waveCtx.moveTo(i,y); else waveCtx.lineTo(i,y); });
  waveCtx.stroke();
}

// ── RAM MONITOR ──────────────────────────────────────────────────────────────
async function pollMemory() {
  try {
    var r = await fetch('/api/system_memory');
    var d = await r.json();
    if (d.available) return;
    var fill = document.getElementById('memFill');
    var pct  = document.getElementById('memPct');
    if (fill) {
      fill.style.width = d.used_pct + '%';
      fill.className = 'mem-bar-fill' + (d.warning ? ' warn' : '');
    }
    if (pct) pct.textContent = d.used_pct + '%';
    var acqRam = document.getElementById('acqRam');
    if (acqRam) acqRam.textContent = d.available_gb + 'GB free';
  } catch(e) {}
}
// Memory polling disabled per user preference

// ── AUTO DETECT SCALE ────────────────────────────────────────────────────────
window.autoDetectScale = async function() {
  var grid = document.getElementById('expGrid');
  var firstCard = grid ? grid.querySelector('.exp-card') : null;
  if (!currentModalId && !firstCard) {
    alert('Upload an experiment first, then use this from the experiment card.');
    return;
  }
  var expId = currentModalId || (firstCard && firstCard.dataset.expId);
  if (!expId) return;
  var btn = document.getElementById('scaleDetectBtn');
  btn.textContent = '⏳ Detecting…';
  btn.disabled = true;
  try {
    var r = await fetch('/api/detect_scale_bar/' + expId);
    var d = await r.json();
    if (d.um_per_px) {
      document.getElementById('pixelScale').value = d.um_per_px.toFixed(4);
      btn.textContent = '✓ Detected: ' + d.um_per_px.toFixed(4) + ' µm/px (' + d.confidence + ')';
    } else {
      btn.textContent = '⚠ Not detected — enter manually';
    }
  } catch(e) {
    btn.textContent = '✗ Detection failed';
  }
  setTimeout(function(){ btn.textContent = '🔍 Auto-detect scale bar'; btn.disabled = false; }, 3000);
};

// ── MAIN DATA MODAL ──────────────────────────────────────────────────────────
window.openModal = function(expId) {
  currentModalId = expId;
  document.getElementById('mainModal').classList.add('active');
  document.getElementById('modalTitle').textContent = 'Loading…';
  document.getElementById('modalPlot').innerHTML = '<p style="color:var(--bio-teal);font-family:var(--mono);padding:20px;font-size:.8rem">Loading…</p>';
  document.getElementById('modalMetrics').innerHTML = '';
  document.getElementById('modalGallery').innerHTML = '';
  document.getElementById('modalRose').innerHTML = '<span style="color:var(--text-mute);font-family:var(--mono);font-size:.75rem">No trajectory data</span>';
  document.getElementById('modalFijiBtn').style.display = 'inline-flex';
  document.getElementById('modalCIBtn').style.display   = 'none';

  loadPlayerFrames(expId, 'modal');

  fetch('/results_json/' + expId)
    .then(function(r){ return r.json(); })
    .then(function(data) {
      document.getElementById('modalTitle').textContent = data.experiment_name || 'Experiment';

      var plotDiv = document.getElementById('modalPlot');
      plotDiv.innerHTML = '';
      if (data.plot_json) {
        try {
          var pd = JSON.parse(data.plot_json);
          Plotly.newPlot(plotDiv, pd.data, pLayout(pd.layout), {responsive:true});
        } catch(e) {
          if (data.plot_b64) plotDiv.innerHTML = '<img src="data:image/png;base64,' + data.plot_b64 + '" style="width:100%">';
        }
      } else if (data.plot_b64) {
        plotDiv.innerHTML = '<img src="data:image/png;base64,' + data.plot_b64 + '" style="width:100%">';
      } else {
        plotDiv.innerHTML = '<p style="color:var(--text-mute);font-family:var(--mono);padding:40px;text-align:center;font-size:.8rem">No plot available</p>';
      }

      if (data.sigmoid_model && data.sigmoid_model !== 'insufficient_data') {
        document.getElementById('modalCIBtn').style.display = 'inline-flex';
      }

      if (data.rose_plot_b64) {
        document.getElementById('modalRose').innerHTML = '<img src="data:image/png;base64,' + data.rose_plot_b64 + '" style="width:100%;border:1px solid rgba(255,255,255,.06)">';
      }

      var tbl = document.getElementById('modalMetrics');
      tbl.innerHTML = '';
      var bioKeys = ['substrate_material','substrate_stiffness_kpa','treatment'];
      var GROUPS = [
        { label:'Gap Closure Kinetics', keys:['initial_area_um2','final_area_um2','final_closure_pct','time_to_25_closure_hr','time_to_50_closure_hr','time_to_75_closure_hr','time_to_90_closure_hr'] },
        { label:'Linear Gap Kinetics', keys:['healing_rate_um2_per_hr','r_squared'] },
        { label:'Sigmoidal Closure Kinetics', keys:['sigmoid_model','sigmoid_asymptote_pct','sigmoid_max_rate_pct_hr','sigmoid_lag_phase_hr','sigmoid_inflection_hr','sigmoid_r_squared'] },
        { label:'Migration Margin Velocity', keys:['left_edge_velocity_um_hr','right_edge_velocity_um_hr','edge_asymmetry_index'] },
        { label:'Margin Roughness (Tortuosity)', keys:['initial_tortuosity','final_tortuosity','edge_smoothing_rate'] },
        { label:'Directed Motility vs Proliferation', keys:['migration_fraction','proliferation_fraction'] },
        { label:'Single-Cell Trajectory Profiling', keys:['num_cells_tracked','mean_velocity_um_min','migration_efficiency_mean','mean_directionality','meander_index','meander_index_mean','mean_displacement_um','msd_alpha','directed_migration_score','persistence_time_hr'] },
        { label:'Biomaterial', keys: bioKeys },
        { label:'Metadata', keys:['num_timepoints','pixel_scale_um_per_px','segmentation_method','flatfield_applied'] }
      ];

      GROUPS.forEach(function(g) {
        var relevant = g.keys.filter(function(k){ return data[k] != null; });
        if (!relevant.length) return;
        var hdr = tbl.insertRow();
        hdr.className = 'm-group-header';
        hdr.innerHTML = '<td colspan="2">' + g.label + '</td>';
        relevant.forEach(function(k) {
          var info = METRIC_INFO[k] || {name:k, unit:''};
          var v = data[k];
          var decimalKeys = ['r_squared','sigmoid_r_squared','migration_efficiency_mean','mean_directionality','migration_fraction','proliferation_fraction','edge_asymmetry_index','msd_alpha','directed_migration_score','meander_index','meander_index_mean'];
          if (typeof v === 'number') {
            if (decimalKeys.indexOf(k) !== -1) v = v.toFixed(3);
            else if (Number.isInteger(v)) v = String(v);
            else v = v.toFixed(2);
          } else if (typeof v === 'boolean') {
            v = v ? 'Yes' : 'No';
          }
          var row = tbl.insertRow();
          if (bioKeys.indexOf(k) !== -1) row.className = 'bio-row';
          row.innerHTML = '<td>' + info.name + '</td><td>' + v + (info.unit ? ' ' + info.unit : '') + '</td>';
        });
      });

      updateLabInsights(data);

      var gal = document.getElementById('modalGallery');
      gal.innerHTML = '';
      var thumbs = data.gallery_thumbs || [];
      thumbs.forEach(function(url) {
        gal.innerHTML += '<a href="' + url + '" target="_blank"><img src="' + url + '" alt="frame" loading="lazy"></a>';
      });
      if (thumbs.length >= 2) {
        document.getElementById('slideLeft').style.backgroundImage  = "url('" + thumbs[0] + "')";
        document.getElementById('slideRight').style.backgroundImage = "url('" + thumbs[thumbs.length-1] + "')";
      }
    })
    .catch(function(e) {
      document.getElementById('modalTitle').textContent = 'Load error';
      console.error(e);
    });
};

window.loadSigmoidCI = async function() {
  if (!currentModalId) return;
  var btn = document.getElementById('modalCIBtn');
  btn.textContent = '⏳';
  btn.disabled = true;
  try {
    var r = await fetch('/api/sigmoid_ci/' + currentModalId);
    var d = await r.json();
    if (d.error) throw new Error(d.error);
    var plotDiv = document.getElementById('modalPlot');
    if (plotDiv && d.traces) Plotly.addTraces(plotDiv, d.traces);
    btn.textContent = '✓ 95% CI Added';
  } catch(e) {
    btn.textContent = 'Error loading CI';
    console.error(e);
  }
  setTimeout(function(){ btn.disabled = false; btn.textContent = '95% CI Band'; }, 3000);
};

window.downloadFijiFromModal = function() {
  if (currentModalId) window.location.href = '/download-fiji/' + currentModalId;
};

var sc = document.getElementById('slideCompare');
var sl = document.getElementById('slideLeft');
var sh = document.querySelector('.slide-handle');
function moveSlider(e) {
  if (!sc || !sl || !sh) return;
  var r = sc.getBoundingClientRect();
  var clientX = e.clientX || (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
  var x = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
  sl.style.width = (x*100) + '%';
  sh.style.left  = (x*100) + '%';
}
if (sc) {
  sc.addEventListener('mousemove', moveSlider);
  sc.addEventListener('touchmove', moveSlider);
}

window.closeModal = function() {
  document.getElementById('mainModal').classList.remove('active');
  playerPlaying = false; clearTimeout(playerTimer);
  currentModalId = null;
};

// ── PUBLICATION & CSV EXPORTS ────────────────────────────────────────────────
window.openPubModal  = function(expId) { currentPubId = expId; document.getElementById('pubModal').classList.add('active'); };
window.closePubModal = function() { document.getElementById('pubModal').classList.remove('active'); currentPubId = null; };

window.downloadPubFigure = function() {
  if (!currentPubId) return;
  var w     = document.getElementById('pubWidth').value;
  var h     = document.getElementById('pubHeight').value;
  var dpi   = document.getElementById('pubDPI').value;
  var style = document.getElementById('pubStyle').value;
  var url   = '/api/publication_figure/' + currentPubId + '?width_mm=' + w + '&height_mm=' + h + '&dpi=' + dpi + '&style=' + style;
  var a = document.createElement('a'); a.href = url; a.click();
};

window.downloadCSV = function(expId) {
  fetch('/results_json/' + expId)
    .then(function(r){ return r.json(); })
    .then(function(data) {
      if (!data.csv_b64) { alert('CSV not available'); return; }
      var blob = new Blob([atob(data.csv_b64)], {type:'text/csv'});
      var a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = (data.experiment_name || expId.replace(/\//g,'_')) + '_timeseries.csv';
      a.click();
    });
};

window.deleteExp = function(e, id, name) {
  e.stopPropagation();
  if (!confirm('⚠ DELETE EXPERIMENT\n\n' + name + '\n\nThis cannot be undone.')) return;
  var card = e.target.closest('.exp-card');
  fetch('/api/delete_experiment', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({result_id:id}) })
    .then(function(r){ return r.json(); })
    .then(function(d) {
      if (d.status==='success' && card) {
        card.style.opacity='0'; card.style.transform='scale(0.95)';
        setTimeout(function(){ card.remove(); }, 300);
      } else alert('Delete failed: ' + (d.error||'Unknown'));
    });
};

// ── COMPARE TAB ──────────────────────────────────────────────────────────────
function loadCompareData() {
  fetch('/api/comparison_data').then(function(r){ return r.json(); }).then(function(data) {
    allExperimentsData = data;
    document.getElementById('compareLoader').style.display='none';
    document.getElementById('compareContent').style.display='grid';

    var list = document.getElementById('compareList');
    list.innerHTML = '<h4>Available Datasets</h4>';
    if (!data.length) {
      list.innerHTML += '<p style="color:var(--text-mute);font-family:var(--mono);font-size:.78rem;margin-top:8px">NO DATA FOUND</p>';
      return;
    }
    data.forEach(function(exp) {
      var d = document.createElement('div');
      d.className = 'chk-item';
      d.innerHTML = '<label><input type="checkbox" class="cmp-chk" value="' + exp.id + '"><span>' + exp.experiment_name + '</span></label>';
      list.appendChild(d);
    });

    var sel = document.getElementById('metricSelect');
    if (sel) {
      sel.innerHTML = '';
      METRICS_COMPARE.forEach(function(k) {
        var info = METRIC_INFO[k] || {name:k, unit:''};
        var o = document.createElement('option');
        o.value = k; o.textContent = info.name + (info.unit ? ' (' + info.unit + ')' : '');
        sel.appendChild(o);
      });
    }
    renderComparePlot();
  }).catch(function() {
    var loader = document.getElementById('compareLoader');
    if (loader) loader.innerHTML = '<p style="color:var(--bio-red);font-family:var(--mono)">LOAD_FAILED</p>';
  });
}

window.renderComparePlot = function() {
  var selected = Array.from(document.querySelectorAll('.cmp-chk:checked')).map(function(c){ return c.value; });
  var key = document.getElementById('metricSelect').value;
  var mi  = METRIC_INFO[key] || {name:key, unit:''};
  var exps = allExperimentsData.filter(function(e){ return selected.indexOf(e.id) !== -1; });
  var div  = document.getElementById('compare-plot');

  if (!exps.length) {
    Plotly.newPlot(div, [], pLayout({title:'SELECT EXPERIMENTS TO COMPARE'}), {responsive:true});
    return;
  }
  var trace = {
    x: exps.map(function(e){ return e.experiment_name; }),
    y: exps.map(function(e){ return e[key]||0; }),
    type: 'bar',
    marker: {color:'#2ecc71', line:{color:'#1abc9c', width:1}}
  };
  
  var layout = pLayout({
    title: mi.name,
    yaxis: {title: mi.name + (mi.unit ? ' (' + mi.unit + ')' : '')},
    xaxis: {tickangle:-40},
    margin:{b:120,t:50,l:60,r:30}
  });
  Plotly.newPlot(div, [trace], layout, {responsive:true});

  if (selected.length >= 2) {
    fetch('/api/compare_stats', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({exp_ids: selected, metric: key})
    }).then(function(r){ return r.json(); }).then(function(d) {
      if (d.valid) {
        var maxY = Math.max.apply(null, exps.map(function(e){ return e[key] || 0; }));
        var offset = maxY * 0.05;
        var pText = d.significance === 'ns' ? 'ns' : d.significance;
        var msg = '<b>' + pText + '</b><br><span style="font-size:9px;color:rgba(255,255,255,0.5)">' + d.test_type + ' p=' + d.p_value.toExponential(2) + '</span>';
        var update = {
          annotations: [{
            x: (exps.length - 1) / 2,
            y: maxY + (offset*3.5),
            xref: 'x', yref: 'y',
            text: msg,
            showarrow: false,
            font: {family: 'DM Mono', size: 14, color: '#f39c12'}
          }],
          shapes: [{
            type: 'path',
            path: 'M 0,' + (maxY + offset) + ' L 0,' + (maxY + offset*2) + ' L ' + (exps.length - 1) + ',' + (maxY + offset*2) + ' L ' + (exps.length - 1) + ',' + (maxY + offset),
            line: {color: 'rgba(243, 156, 18, 0.7)', width: 1.5}
          }]
        };
        Plotly.relayout(div, update);
      }
    }).catch(function(e){ console.error('Stats error:', e); });
  }
};

// ── PUBCHEM INTEGRATION ──────────────────────────────────────────────────────
window.fetchPubChemData = function(materialName) {
  var panel   = document.getElementById('pubchemPanel');
  var content = document.getElementById('pubchemContent');
  if (!materialName || materialName.indexOf('Steel') !== -1 || materialName.indexOf('Ceramic') !== -1 || materialName.indexOf('Glass') !== -1) {
    panel.style.display = 'none'; return;
  }
  panel.style.display = 'block';
  content.className = 'pubchem-loader';
  content.innerHTML = 'Querying PubChem REST API for [' + materialName + '] <span>...</span>';
  fetch('/api/pubchem/' + encodeURIComponent(materialName))
    .then(function(r){ return r.json(); })
    .then(function(data) {
      if (data.status === 'success') {
        content.className = 'pubchem-data';
        content.innerHTML =
          '<div class="pubchem-key">Standard Name:</div><div class="pubchem-value" style="color:var(--text)">' + data.title + '</div>' +
          '<div class="pubchem-key">Molecular Weight:</div><div class="pubchem-value">' + data.mw + ' g/mol</div>' +
          '<div class="pubchem-key">SMILES string:</div><div class="pubchem-value" style="font-size:0.65rem">' + data.smiles + '</div>';
      } else {
        content.className = 'pubchem-data';
        content.innerHTML = '<div class="pubchem-key" style="color:var(--bio-amber)">Status:</div><div class="pubchem-value" style="color:var(--text-mute)">' + data.message + '</div>';
      }
    })
    .catch(function() {
      content.innerHTML = '<span style="color:var(--bio-red)">Failed to connect to backend PubChem API.</span>';
    });
};

// ── IN SILICO SCAFFOLD DESIGNER ──────────────────────────────────────────────
var selectedMaterial = 'PHEMA Hydrogel';
var sdInitialised = false;

function initScaffoldDesigner() {
  if (sdInitialised) return;
  sdInitialised = true;
  var materials = ["Stainless Steel 316L", "Alumina Ceramic", "PHEMA Hydrogel", "PEGDA", "Collagen I", "Alginate"];
  var cardRow = document.getElementById('materialCards');
  cardRow.innerHTML = '';
  materials.forEach(function(name) {
    var info = SCAFFOLD_DB[name] || {};
    var card = document.createElement('div');
    card.className = 'mat-tile' + (name === selectedMaterial ? ' selected' : '');
    card.innerHTML = '<div class="mat-swatch" style="background:' + (info.color || '#333') + '"></div><div class="mat-name">' + name + '</div>';
    card.onclick = function() {
      document.querySelectorAll('.mat-tile').forEach(function(c){ c.classList.remove('selected'); });
      card.classList.add('selected');
      selectedMaterial = name;
      updateStiffnessDisplay(document.getElementById('sdStiffness'));
    };
    cardRow.appendChild(card);
  });
  updateStiffnessDisplay(document.getElementById('sdStiffness'));
}

window.updateStiffnessDisplay = function(el) {
  var logVal = parseFloat(el.value);
  var kpa    = Math.pow(10, logVal);
  document.getElementById('sdStiffnessVal').textContent = kpa < 1 ? (kpa.toFixed(2) + ' kPa') : kpa < 100 ? (kpa.toFixed(1) + ' kPa') : (Math.round(kpa) + ' kPa');
  var region = '';
  if (kpa < 1) region = 'Brain/Neural';
  else if (kpa < 8) region = 'Soft Tissue';
  else if (kpa < 25) region = 'Muscle-like';
  else if (kpa < 1000) region = 'Cartilage-like';
  else region = 'Bone/Glass-like';
  document.getElementById('sdStiffnessRegion').textContent = region;
  var pct = ((logVal - (-1)) / (4 - (-1))) * 100;
  document.getElementById('stiffnessPointer').style.left = Math.max(0, Math.min(100, pct)) + '%';
};

window.runScaffoldDesign = async function() {
  var btn = document.getElementById('sdBtn');
  btn.disabled = true;
  btn.innerHTML = '<span>⏳ COMPUTING...</span>';
  document.getElementById('sdPlaceholder').style.display = 'none';
  document.getElementById('sdResultContent').style.display = 'block';
  try {
    var stiffness = Math.pow(10, parseFloat(document.getElementById('sdStiffness').value));
    var scaffoldRes = await fetch('/api/scaffold_design', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        material: selectedMaterial,
        stiffness_kpa: stiffness,
        crosslink_density: parseFloat(document.getElementById('sdCrosslink').value),
        cell_type: document.getElementById('sdCellType').value
      })
    });
    var d = await scaffoldRes.json();
    displayScaffoldResult(d);

    var payload = {
      material: selectedMaterial,
      stiffness_kpa: stiffness,
      crosslink_density: parseFloat(document.getElementById('sdCrosslink').value),
      cell_type: document.getElementById('sdCellType').value
    };

    var res1 = await fetch('/api/insilico/durotaxis_traction', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    var data1 = await res1.json();
    Plotly.newPlot('durotaxis-plot', data1.data, pLayout(data1.layout), {responsive:true});

    var res2 = await fetch('/api/insilico/immune_trajectory', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    var data2 = await res2.json();
    Plotly.newPlot('immune-plot', data2.data, pLayout(data2.layout), {responsive:true});

  } catch(e) {
    alert('Simulation failed: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>⚡ Run Prediction</span>';
  }
};

function displayScaffoldResult(d) {
  if (d.error) { alert('Error: ' + d.error); return; }
  var info = SCAFFOLD_DB[selectedMaterial] || {};
  document.getElementById('sdMatPreview').textContent = selectedMaterial;
  document.getElementById('sdMatPreview').style.background = info.color || '#333';
  document.getElementById('sdMatPreview').style.color = '#000';
  document.getElementById('sdRate').textContent = d.healing_rate_um2_hr.toLocaleString();
  document.getElementById('sdVel').textContent = d.velocity_um_min.toFixed(2);
  document.getElementById('sdClosure24').textContent = d.closure_pct_24h.toFixed(1) + '%';
  document.getElementById('sdPorosity').textContent = d.effective_porosity + '%';
  document.getElementById('sdDeg').textContent = d.degradation_days ? ('~' + d.degradation_days + 'd') : 'Non-degradable';
  document.getElementById('sdAdh').textContent = d.cell_adhesion;
  document.getElementById('sdUse').textContent = d.common_use;
  var regimeEl = document.getElementById('sdRegime');
  regimeEl.textContent = d.migration_mode;
  regimeEl.className = 'scaffold-note amber';
  var recEl = document.getElementById('sdRec');
  recEl.textContent = d.recommendation;
  recEl.className = 'scaffold-note green';
  document.getElementById('sdGaugeFill').style.width = Math.min(100, (d.healing_rate_um2_hr / 4000) * 100) + '%';
}

// ── KEYBOARD & INIT ──────────────────────────────────────────────────────────
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') { closeModal(); closePubModal(); closePlayerModal(); }
});
document.addEventListener('DOMContentLoaded', function() {
  console.log('WoundTrack AI v4 — Biology Edition online');
  if (document.querySelector('.s-tab.active') && document.querySelector('.s-tab.active').textContent.indexOf('Analytics') !== -1) {
    renderStatsPlots();
  }
});

// ── AI MODAL ─────────────────────────────────────────────────────────────────
let _aiCurrentExpId = null;
let _aiCurrentMode  = 'interpret';

window.openAIPanel = function(expId) {
  _aiCurrentExpId = expId;
  _aiCurrentMode  = 'interpret';
  // Reset all panes
  ['interpret','anomaly','predict'].forEach(m => {
    const el = document.getElementById('ai-out-' + m);
    if (el) el.classList.remove('loading');
  });
  document.getElementById('aiModal').classList.add('active');
  // Show interpret pane
  switchAITab(document.querySelector('.ai-tab'), 'interpret');
};

window.closeAIModal = function() {
  document.getElementById('aiModal').classList.remove('active');
};

window.switchAITab = function(btn, mode) {
  _aiCurrentMode = mode;
  document.querySelectorAll('.ai-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  ['interpret','anomaly','predict','score','novel'].forEach(m => {
    const p = document.getElementById('ai-pane-' + m);
    if (p) p.style.display = m === mode ? '' : 'none';
  });
};

window.runAI = async function(mode) {
  if (!_aiCurrentExpId) return;
  const outEl = document.getElementById('ai-out-' + mode);
  if (!outEl) return;
  outEl.textContent = 'Analysing with Claude AI…';
  outEl.classList.add('loading');

  const btns = document.querySelectorAll('.ai-run-btn');
  btns.forEach(b => b.disabled = true);

  try {
    const res  = await fetch('/api/ai_interpret', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({exp_id: _aiCurrentExpId, mode: mode})
    });
    const data = await res.json();
    outEl.classList.remove('loading');
    if (data.error) {
      outEl.textContent = '⚠ Error: ' + data.error;
    } else {
      outEl.textContent = data.interpretation || 'No interpretation returned.';
    }
  } catch(e) {
    outEl.classList.remove('loading');
    outEl.textContent = '⚠ Network error: ' + e.message;
  } finally {
    btns.forEach(b => b.disabled = false);
  }
};

window.computeWTS = async function() {
  if (!_aiCurrentExpId) return;
  const card    = document.getElementById('wtsCard');
  const interp  = document.getElementById('wtsInterpret');
  const runBtn  = document.querySelector('#ai-pane-score .ai-run-btn');

  if (runBtn) runBtn.disabled = true;
  if (card)   card.style.display = 'none';
  if (interp) { interp.style.display = 'block'; interp.textContent = 'Computing WoundTrack Score™…'; interp.classList.add('loading'); }

  try {
    const res  = await fetch('/api/ai_composite_score', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({exp_id: _aiCurrentExpId})
    });
    const data = await res.json();

    if (data.error) {
      interp.textContent = '⚠ Error: ' + data.error;
      interp.classList.remove('loading');
      return;
    }

    // Populate score card
    document.getElementById('wtsTotal').textContent = data.wts_total;
    document.getElementById('wtsGrade').textContent = data.grade;

    const setBar = (id, val) => {
      const pct = (val / 25) * 100;
      document.getElementById('wtsBar-' + id).style.width = pct + '%';
      document.getElementById('wtsVal-' + id).textContent = val + '/25';
    };
    setBar('k',  data.kinetics_score);
    setBar('m',  data.migration_score);
    setBar('mo', data.morpho_score);
    setBar('c',  data.confidence_score);

    card.style.display = 'block';
    interp.classList.remove('loading');
    interp.textContent = data.interpretation;

  } catch(e) {
    interp.classList.remove('loading');
    interp.textContent = '⚠ Network error: ' + e.message;
  } finally {
    if (runBtn) runBtn.disabled = false;
  }
};

// ── NOVEL METRICS JS ─────────────────────────────────────────────────────────
const NOVEL_LABELS = {
  final_closure_pct:       'Closure %',
  sigmoid_max_rate_pct_hr: 'Peak Rate',
  sigmoid_lag_phase_hr:    'Lag Phase',
  sigmoid_r_squared:       'Sigmoid R²',
  msd_alpha:               'MSD α',
  directed_migration_score:'Directed Migr.',
  edge_asymmetry_index:    'Edge Asym.',
  initial_tortuosity:      'Tortuosity',
  migration_fraction:      'Migr. Fraction',
  mean_velocity_um_min:    'Velocity',
  regularity_score:        'Regularity',
  wavefront_coherence:     'Wave Coherence',
};

window.runNovel = async function(type) {
  if (!_aiCurrentExpId) return;
  const outEl = document.getElementById('novel-out-' + type);
  if (outEl) { outEl.innerHTML = '<div class="novel-placeholder">Computing…</div>'; }

  try {
    const res  = await fetch('/api/novel/' + type + '/' + _aiCurrentExpId);
    const data = await res.json();
    if (data.error) {
      if (outEl) outEl.innerHTML = '<div class="novel-placeholder" style="color:#e74c3c">⚠ ' + data.error + '</div>';
      return;
    }
    renderNovelOutput(type, data, outEl);
  } catch(e) {
    if (outEl) outEl.innerHTML = '<div class="novel-placeholder" style="color:#e74c3c">⚠ ' + e.message + '</div>';
  }
};

function renderNovelOutput(type, d, el) {
  if (!el) return;

  if (type === 'wavefront') {
    const regime = d.healing_regime || '—';
    const regimeColor = {
      'wave-driven': '#2ecc71', 'diffusive': '#3498db',
      'arrested': '#e74c3c', 'stochastic': '#e67e22', 'insufficient_data': '#7f8c8d'
    }[regime] || '#aaa';
    el.innerHTML = '<div class="novel-metric-grid">' +
      kv('Wave Speed', (d.wave_speed_um_hr != null ? d.wave_speed_um_hr + ' µm/hr' : '—')) +
      kv('Decay λ', (d.decay_constant_hr != null ? d.decay_constant_hr + ' /hr' : '—')) +
      kv('Coherence R²', (d.wavefront_coherence != null ? d.wavefront_coherence : '—')) +
      kv('Period', (d.wavefront_period_hr != null ? d.wavefront_period_hr + ' hr' : 'Aperiodic')) +
      kv('Predicted Closure', (d.predicted_closure_hr != null ? d.predicted_closure_hr + ' hr' : '—')) +
      '<div class="novel-kv"><div class="nk">Healing Regime</div>' +
      '<div class="nv regime" style="color:' + regimeColor + '">' + regime + '</div></div>' +
      '</div>';
  }

  else if (type === 'entropy') {
    const H = d.healing_entropy != null ? d.healing_entropy.toFixed(3) : '—';
    const reg = d.regularity_score != null ? (d.regularity_score * 100).toFixed(1) + '%' : '—';
    const cv  = d.rate_cv != null ? d.rate_cv.toFixed(3) : '—';
    el.innerHTML =
      '<div class="entropy-display">' +
        '<div class="entropy-big">' + H + '</div>' +
        '<div class="entropy-label">bits<br>Shannon entropy</div>' +
      '</div>' +
      '<div class="novel-metric-grid" style="grid-template-columns:1fr 1fr">' +
        kv('Regularity Score', reg) +
        kv('Rate CV (σ/μ)', cv) +
      '</div>' +
      '<div style="font-family:var(--mono);font-size:.65rem;color:#1abc9c;margin-top:8px;line-height:1.5">' +
        (d.entropy_interpretation || '') + '</div>';
  }

  else if (type === 'phase') {
    const ph = d.phase || '—';
    const em = d.emoji || '';
    const conf = d.confidence != null ? (d.confidence * 100).toFixed(0) : 0;
    let scoresHtml = '';
    if (d.scores) {
      const sorted = Object.entries(d.scores).sort((a,b) => b[1]-a[1]);
      scoresHtml = sorted.map(([p, s]) =>
        '<div class="fp-bar-row"><span class="fp-bar-label">' + p + '</span>' +
        '<div class="fp-bar-track"><div class="fp-bar-fill" style="width:' + (s*100).toFixed(0) + '%;background:linear-gradient(90deg,#e67e22,#f39c12)"></div></div>' +
        '<span class="fp-bar-val">' + (s*100).toFixed(0) + '%</span></div>'
      ).join('');
    }
    el.innerHTML =
      '<div class="phase-badge">' + em + ' ' + ph + '</div>' +
      '<div class="phase-conf-bar"><div class="phase-conf-fill" style="width:' + conf + '%"></div></div>' +
      '<div style="font-family:var(--mono);font-size:.65rem;color:var(--text-mute);margin-bottom:10px">' +
        (d.description || '') + ' &nbsp;(' + conf + '% confidence)</div>' +
      '<div class="fp-bar-grid">' + scoresHtml + '</div>';
  }

  else if (type === 'fingerprint') {
    const dims = d.dimensions || [];
    const raw  = d.raw_normalised || [];
    const comp = d.completeness != null ? (d.completeness * 100).toFixed(0) + '%' : '—';
    let bars = dims.map((dim, i) =>
      '<div class="fp-bar-row">' +
        '<span class="fp-bar-label">' + (NOVEL_LABELS[dim] || dim) + '</span>' +
        '<div class="fp-bar-track"><div class="fp-bar-fill" style="width:' + (raw[i]*100).toFixed(0) + '%\"></div></div>' +
        '<span class="fp-bar-val">' + (raw[i] != null ? (raw[i]*100).toFixed(0) : '—') + '</span>' +
      '</div>'
    ).join('');
    el.innerHTML =
      '<div style=\"font-family:var(--mono);font-size:.65rem;color:var(--text-mute);margin-bottom:8px\">' +
        'Completeness: <span style=\"color:#3498db\">' + comp + '</span> &nbsp;|&nbsp; ' +
        'Vector norm: <span style=\"color:#3498db\">' + (d.vector_norm || '—') + '</span>' +
        (d.missing_dims && d.missing_dims.length ? ' &nbsp;| Missing: ' + d.missing_dims.join(', ') : '') +
      '</div>' +
      '<div class=\"fp-bar-grid\">' + bars + '</div>';
  }
}

function kv(label, value) {
  return '<div class="novel-kv"><div class="nk">' + label + '</div><div class="nv">' + value + '</div></div>';
}

