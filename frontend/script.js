// ═══════════════════════════════════════════
//  DECEPTIQ v2.0 — MAIN SCRIPT
//  Live webcam + video upload analysis
// ═══════════════════════════════════════════

const API_URL = 'http://localhost:5000';

// Chart instances
let resultTimelineChart = null;
let resultDistributionChart = null;
let resultGaugeChart = null;
let liveChartInstance = null;

// Webcam state
let webcamStream = null;
let analysisInterval = null;
let liveFrameCount = 0;
let liveSpikeCount = 0;
let liveRiskHistory = [];
let liveEmotionHistory = {
  fear: [], angry: [], disgust: [], happy: [], neutral: [], sad: [], surprise: []
};
const MAX_HISTORY = 30;

// ── SCREEN NAVIGATION ────────────────────────
function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

function goHome() {
  stopWebcam();
  showScreen('homeScreen');
}

// ── FILE UPLOAD ──────────────────────────────
document.getElementById('videoFileInput').addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (file) uploadAndAnalyze(file);
});

async function uploadAndAnalyze(file) {
  showScreen('loadingScreen');
  animateLoadingSteps();

  const formData = new FormData();
  formData.append('video', file);

  try {
    const response = await fetch(`${API_URL}/analyze`, {
      method: 'POST',
      body: formData
    });

    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Analysis failed');
    displayResults(data);

  } catch (err) {
    alert('Error: ' + err.message + '\n\nMake sure Flask backend is running!');
    goHome();
  }
}

// ── DEMO MODE ────────────────────────────────
async function runDemo() {
  showScreen('loadingScreen');
  animateLoadingSteps();

  try {
    const response = await fetch(`${API_URL}/analyze-demo`);
    const data = await response.json();
    if (!data.success) throw new Error('Demo failed');
    displayResults(data);
  } catch (err) {
    alert('Cannot connect to backend!\n\nRun: python app.py in your backend folder');
    goHome();
  }
}

// ── LOADING ANIMATION ────────────────────────
function animateLoadingSteps() {
  const steps = ['lstep1', 'lstep2', 'lstep3', 'lstep4'];
  steps.forEach(s => {
    const el = document.getElementById(s);
    el.classList.remove('active', 'done');
  });

  let i = 0;
  document.getElementById(steps[0]).classList.add('active');

  const iv = setInterval(() => {
    if (i < steps.length - 1) {
      document.getElementById(steps[i]).classList.remove('active');
      document.getElementById(steps[i]).classList.add('done');
      i++;
      document.getElementById(steps[i]).classList.add('active');
    } else {
      clearInterval(iv);
    }
  }, 1800);
}

// ── DISPLAY RESULTS ──────────────────────────
function displayResults(data) {
  showScreen('resultsScreen');

  const { risk, video_info, timeline } = data;

  // Score
  animateNumber('vbScore', risk.score);
  const lvl = document.getElementById('vbLevel');
  lvl.textContent = risk.level;
  lvl.style.color = risk.color;

  // Stats
  document.getElementById('vbDuration').textContent = `${video_info.duration}s`;
  document.getElementById('vbFrames').textContent = video_info.frames_analyzed;
  document.getElementById('vbFps').textContent = parseFloat(video_info.fps).toFixed(1);

  // Indicators
  renderResultIndicators(risk.indicators);

  // Charts
  setTimeout(() => {
    renderResultTimeline(timeline);
    renderResultDistribution(timeline);
    renderResultGauge(risk.score, risk.color);
  }, 100);
}

function renderResultIndicators(indicators) {
  const el = document.getElementById('resultIndicators');

  if (indicators.length === 0) {
    el.innerHTML = `<div class="no-indicators">✅ No significant deception indicators detected.</div>`;
    return;
  }

  el.innerHTML = indicators.map(ind => `
    <div class="indicator-card ${ind.severity}">
      <div class="ind-severity">${ind.severity.toUpperCase()}</div>
      <div class="ind-content">
        <div class="ind-name">${ind.name}</div>
        <div class="ind-desc">${ind.description}</div>
      </div>
    </div>
  `).join('');
}

function renderResultTimeline(timeline) {
  const ctx = document.getElementById('resultTimeline').getContext('2d');
  if (resultTimelineChart) resultTimelineChart.destroy();

  const labels = timeline.map(f => `${f.timestamp}s`);
  const emotionColors = {
    fear:     '#bf5af2',
    angry:    '#ff2d55',
    disgust:  '#ff9f0a',
    happy:    '#00ff9d',
    neutral:  '#4a6580',
    sad:      '#0a84ff',
    surprise: '#00e5ff',
  };

  const datasets = Object.entries(emotionColors).map(([emotion, color]) => ({
    label: emotion.charAt(0).toUpperCase() + emotion.slice(1),
    data: timeline.map(f => f.emotions[emotion] || 0),
    borderColor: color,
    backgroundColor: color + '15',
    borderWidth: 2,
    fill: false,
    tension: 0.4,
    pointRadius: 2,
  }));

  resultTimelineChart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#4a6580', font: { family: 'Space Mono', size: 10 }, boxWidth: 12 } },
        tooltip: { backgroundColor: '#080f18', titleColor: '#00e5ff', bodyColor: '#cdd8e3', borderColor: '#112236', borderWidth: 1 }
      },
      scales: {
        x: { ticks: { color: '#4a6580', font: { size: 10 } }, grid: { color: '#0c1624' } },
        y: { min: 0, max: 100, ticks: { color: '#4a6580', font: { size: 10 } }, grid: { color: '#0c1624' } }
      }
    }
  });
}

function renderResultDistribution(timeline) {
  const ctx = document.getElementById('resultDistribution').getContext('2d');
  if (resultDistributionChart) resultDistributionChart.destroy();

  const counts = {};
  timeline.forEach(f => {
    const d = f.dominant || 'none';
    counts[d] = (counts[d] || 0) + 1;
  });

  const colorMap = {
    neutral: '#4a6580', happy: '#00ff9d', sad: '#0a84ff',
    angry: '#ff2d55', fear: '#bf5af2', disgust: '#ff9f0a',
    surprise: '#00e5ff', none: '#112236'
  };

  const labels = Object.keys(counts);
  resultDistributionChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: Object.values(counts),
        backgroundColor: labels.map(l => colorMap[l] || '#888'),
        borderColor: '#03070d',
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#4a6580', font: { family: 'Space Mono', size: 10 }, boxWidth: 10 }
        }
      }
    }
  });
}

function renderResultGauge(score, color) {
  const ctx = document.getElementById('resultGauge').getContext('2d');
  if (resultGaugeChart) resultGaugeChart.destroy();

  resultGaugeChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [score / 100, 1 - score / 100],
        backgroundColor: [color, '#0c1624'],
        borderWidth: 0,
        circumference: 180,
        rotation: 270,
      }]
    },
    options: {
      responsive: true,
      cutout: '78%',
      plugins: { legend: { display: false }, tooltip: { enabled: false } }
    }
  });
}

// ── LIVE WEBCAM MODE ─────────────────────────
function startLiveMode() {
  showScreen('liveScreen');
  initLiveChart();
  resetLiveStats();
}

function initLiveChart() {
  const ctx = document.getElementById('liveChart').getContext('2d');
  if (liveChartInstance) liveChartInstance.destroy();

  liveChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: Array(MAX_HISTORY).fill(''),
      datasets: [
        {
          label: 'Angry',
          data: Array(MAX_HISTORY).fill(0),
          borderColor: '#ff2d55',
          borderWidth: 2,
          fill: false,
          tension: 0.4,
          pointRadius: 0,
        },
        {
          label: 'Fear',
          data: Array(MAX_HISTORY).fill(0),
          borderColor: '#bf5af2',
          borderWidth: 2,
          fill: false,
          tension: 0.4,
          pointRadius: 0,
        },
        {
          label: 'Disgust',
          data: Array(MAX_HISTORY).fill(0),
          borderColor: '#ff9f0a',
          borderWidth: 2,
          fill: false,
          tension: 0.4,
          pointRadius: 0,
        }
      ]
    },
    options: {
      responsive: true,
      animation: false,
      plugins: {
        legend: { labels: { color: '#4a6580', font: { family: 'Space Mono', size: 9 }, boxWidth: 10 } }
      },
      scales: {
        x: { display: false },
        y: {
          min: 0, max: 100,
          ticks: { color: '#4a6580', font: { size: 9 } },
          grid: { color: '#0c1624' }
        }
      }
    }
  });
}

async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: false
    });

    const video = document.getElementById('webcamVideo');
    video.srcObject = webcamStream;

    document.getElementById('startBtn').classList.add('hidden');
    document.getElementById('stopBtn').classList.remove('hidden');
    document.getElementById('statusText').textContent = 'ANALYZING LIVE...';
    document.getElementById('recordingIndicator').classList.add('active');

    // Start analyzing every 1.5 seconds
    analysisInterval = setInterval(captureAndAnalyze, 1500);

  } catch (err) {
    alert('Camera access denied!\n\nPlease allow camera access in your browser and try again.');
  }
}

function stopWebcam() {
  if (analysisInterval) {
    clearInterval(analysisInterval);
    analysisInterval = null;
  }
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }

  document.getElementById('startBtn').classList.remove('hidden');
  document.getElementById('stopBtn').classList.add('hidden');
  document.getElementById('statusText').textContent = 'STOPPED';
  document.getElementById('recordingIndicator').classList.remove('active');

  resetLiveStats();
}

async function captureAndAnalyze() {
  const video = document.getElementById('webcamVideo');
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 480;
  const ctx = canvas.getContext('2d');

  // Draw current frame (mirror it back for correct orientation)
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Convert to blob and send to backend
  canvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append('frame', blob, 'frame.jpg');

    try {
      const response = await fetch(`${API_URL}/analyze-frame`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) return;

      const data = await response.json();
      if (data.success) {
        updateLiveDisplay(data);
      }

    } catch (err) {
      // Silent fail for live mode
    }
  }, 'image/jpeg', 0.8);
}

function updateLiveDisplay(data) {
  const { emotions, dominant, face_detected, deception_score, verdict } = data;

  liveFrameCount++;
  document.getElementById('framesCount').textContent = liveFrameCount;

  if (!face_detected) {
    document.getElementById('statusText').textContent = 'NO FACE DETECTED';
    return;
  }

  document.getElementById('statusText').textContent = 'FACE LOCKED ON';

  // Update emotion bars
  const emotionOrder = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'];
  emotionOrder.forEach(emotion => {
    const val = emotions[emotion] || 0;
    const bar = document.getElementById(`bar-${emotion}`);
    const valEl = document.getElementById(`val-${emotion}`);
    if (bar) bar.style.width = `${val}%`;
    if (valEl) valEl.textContent = `${Math.round(val)}%`;
  });

  // Update dominant emotion
  document.getElementById('dominantEmotion').textContent = dominant.toUpperCase();

  // Update live chart
  updateLiveChart(emotions);

  // Update risk
  const score = deception_score || 0;
  liveRiskHistory.push(score);
  if (liveRiskHistory.length > MAX_HISTORY) liveRiskHistory.shift();

  const avgScore = Math.round(liveRiskHistory.reduce((a, b) => a + b, 0) / liveRiskHistory.length);
  document.getElementById('riskFill').style.width = `${avgScore}%`;
  document.getElementById('riskScoreLive').textContent = avgScore;

  // Spike detection
  if (score > 50) {
    liveSpikeCount++;
    document.getElementById('spikesCount').textContent = liveSpikeCount;
  }

  // Risk level
  let level, color;
  if (avgScore >= 70) { level = 'HIGH RISK'; color = '#ff2d55'; }
  else if (avgScore >= 40) { level = 'MODERATE RISK'; color = '#ffd60a'; }
  else if (avgScore >= 15) { level = 'LOW RISK'; color = '#0a84ff'; }
  else { level = 'MINIMAL RISK'; color = '#00ff9d'; }

  document.getElementById('riskLevelLive').textContent = level;
  document.getElementById('riskLevelLive').style.color = color;
  document.getElementById('riskScoreLive').style.color = color;

  // VERDICT
  const verdictEl = document.getElementById('verdictResult');
  const confEl = document.getElementById('verdictConfidence');

  if (liveFrameCount < 5) {
    verdictEl.textContent = 'CALIBRATING';
    verdictEl.style.color = '#4a6580';
    confEl.textContent = `Collecting baseline... (${liveFrameCount}/5 frames)`;
  } else if (verdict === 'deceptive') {
    verdictEl.textContent = '⚠ DECEPTIVE';
    verdictEl.style.color = '#ff2d55';
    confEl.textContent = `Risk score: ${avgScore}/100 — Emotional inconsistency detected`;
  } else if (verdict === 'truthful') {
    verdictEl.textContent = '✓ TRUTHFUL';
    verdictEl.style.color = '#00ff9d';
    confEl.textContent = `Risk score: ${avgScore}/100 — Consistent emotional pattern`;
  } else {
    verdictEl.textContent = '? UNCERTAIN';
    verdictEl.style.color = '#ffd60a';
    confEl.textContent = `Risk score: ${avgScore}/100 — Mixed signals detected`;
  }

  // Draw face box on overlay canvas
  if (data.face_box) {
    drawFaceBox(data.face_box, dominant);
  }
}

function updateLiveChart(emotions) {
  if (!liveChartInstance) return;

  // Add new data
  liveEmotionHistory.angry.push(emotions.angry || 0);
  liveEmotionHistory.fear.push(emotions.fear || 0);
  liveEmotionHistory.disgust.push(emotions.disgust || 0);

  // Trim to max length
  ['angry', 'fear', 'disgust'].forEach(e => {
    if (liveEmotionHistory[e].length > MAX_HISTORY) {
      liveEmotionHistory[e].shift();
    }
  });

  liveChartInstance.data.datasets[0].data = [...liveEmotionHistory.angry];
  liveChartInstance.data.datasets[1].data = [...liveEmotionHistory.fear];
  liveChartInstance.data.datasets[2].data = [...liveEmotionHistory.disgust];
  liveChartInstance.update('none');
}

function drawFaceBox(box, emotion) {
  const canvas = document.getElementById('overlayCanvas');
  const video = document.getElementById('webcamVideo');
  const ctx = canvas.getContext('2d');

  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const emotionColors = {
    happy: '#00ff9d', neutral: '#00e5ff', sad: '#0a84ff',
    angry: '#ff2d55', fear: '#bf5af2', disgust: '#ff9f0a', surprise: '#ffd60a'
  };

  const color = emotionColors[emotion] || '#00e5ff';
  const { x, y, w, h } = box;

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.shadowColor = color;
  ctx.shadowBlur = 10;

  // Draw corner markers only
  const cs = 20;
  ctx.beginPath();
  // TL
  ctx.moveTo(x, y + cs); ctx.lineTo(x, y); ctx.lineTo(x + cs, y);
  // TR
  ctx.moveTo(x + w - cs, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + cs);
  // BL
  ctx.moveTo(x, y + h - cs); ctx.lineTo(x, y + h); ctx.lineTo(x + cs, y + h);
  // BR
  ctx.moveTo(x + w - cs, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - cs);
  ctx.stroke();

  // Emotion label
  ctx.fillStyle = color;
  ctx.shadowBlur = 0;
  ctx.font = '700 12px Space Mono, monospace';
  ctx.fillText(emotion.toUpperCase(), x, y - 8);
}

function resetLiveStats() {
  liveFrameCount = 0;
  liveSpikeCount = 0;
  liveRiskHistory = [];
  liveEmotionHistory = {
    fear: [], angry: [], disgust: [], happy: [], neutral: [], sad: [], surprise: []
  };

  document.getElementById('framesCount').textContent = '0';
  document.getElementById('spikesCount').textContent = '0';
  document.getElementById('dominantEmotion').textContent = '—';
  document.getElementById('riskFill').style.width = '0%';
  document.getElementById('riskScoreLive').textContent = '0';
  document.getElementById('riskLevelLive').textContent = 'Collecting data...';
  document.getElementById('verdictResult').textContent = '—';
  document.getElementById('verdictConfidence').textContent = 'Waiting for data...';
  document.getElementById('statusText').textContent = 'READY';

  const emotions = ['neutral','happy','sad','angry','fear','disgust','surprise'];
  emotions.forEach(e => {
    const bar = document.getElementById(`bar-${e}`);
    const val = document.getElementById(`val-${e}`);
    if (bar) bar.style.width = '0%';
    if (val) val.textContent = '0%';
  });
}

// ── ANIMATE NUMBER ───────────────────────────
function animateNumber(id, target) {
  const el = document.getElementById(id);
  let current = 0;
  const step = target / 50;
  const iv = setInterval(() => {
    current = Math.min(current + step, target);
    el.textContent = Math.round(current);
    if (current >= target) clearInterval(iv);
  }, 25);
}