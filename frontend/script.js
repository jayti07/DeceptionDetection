function resolveApiBase() {
    if (window.location.protocol.startsWith("http") && window.location.hostname) {
        return `http://${window.location.hostname}:5000`;
    }
    return "http://127.0.0.1:5000";
}

const API = resolveApiBase();
const EMPTY_EMOTIONS = {
    angry: 0,
    disgust: 0,
    fear: 0,
    happy: 0,
    sad: 0,
    surprise: 0,
    neutral: 0
};

const emotionColors = {
    angry: "#ef4444",
    disgust: "#84cc16",
    fear: "#a855f7",
    happy: "#eab308",
    sad: "#3b82f6",
    surprise: "#f97316",
    neutral: "#94a3b8"
};

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayContext = overlay.getContext("2d");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");

let stream = null;
let liveInterval = null;
let liveChartInstance = null;
let timelineChartInstance = null;
let liveRequestInFlight = false;
let liveSessionId = createSessionId();

function createSessionId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
        return window.crypto.randomUUID();
    }
    return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function formatLabel(value) {
    return value
        .replace(/-/g, " ")
        .replace(/\b\w/g, char => char.toUpperCase());
}

function setVerdictStyle(element, verdict) {
    element.style.color = "#e2e8f0";
    if (verdict === "deceptive") {
        element.style.color = "#ef4444";
    } else if (verdict === "uncertain") {
        element.style.color = "#eab308";
    } else if (verdict === "truthful") {
        element.style.color = "#10b981";
    }
}

function setLiveStatus(message, isError = false) {
    const status = document.getElementById("liveStatus");
    status.textContent = message;
    status.classList.toggle("error-text", isError);
}

function setUploadStatus(message, isError = false) {
    const status = document.getElementById("uploadStatus");
    status.textContent = message;
    status.classList.toggle("error-text", isError);
}

function setUploadError(message) {
    const uploadError = document.getElementById("uploadError");
    if (!message) {
        uploadError.textContent = "";
        uploadError.classList.add("hidden");
        return;
    }

    uploadError.textContent = message;
    uploadError.classList.remove("hidden");
}

function resetLiveMetrics() {
    document.getElementById("liveEmotion").textContent = "-";
    document.getElementById("liveScore").textContent = "0";
    const verdict = document.getElementById("liveVerdict");
    verdict.textContent = "-";
    verdict.style.color = "#e2e8f0";
    drawFaceBox(null, "truthful");
    updateLiveBarChart(EMPTY_EMOTIONS);
}

function updateLiveMetrics(data) {
    document.getElementById("liveEmotion").textContent = formatLabel(data.dominant);
    document.getElementById("liveScore").textContent = String(data.deception_score);
    const verdict = document.getElementById("liveVerdict");
    verdict.textContent = formatLabel(data.verdict);
    setVerdictStyle(verdict, data.verdict);
}

function renderNoFaceState() {
    document.getElementById("liveEmotion").textContent = "No Face";
    document.getElementById("liveScore").textContent = "0";
    const verdict = document.getElementById("liveVerdict");
    verdict.textContent = "Align Face";
    verdict.style.color = "#94a3b8";
    drawFaceBox(null, "truthful");
    updateLiveBarChart(EMPTY_EMOTIONS);
}

async function fetchJson(url, options) {
    const response = await fetch(url, options);
    const contentType = response.headers.get("content-type") || "";

    if (contentType.includes("application/json")) {
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error || `Request failed with status ${response.status}.`);
        }
        return payload;
    }

    const text = await response.text();
    if (!response.ok) {
        throw new Error(text || `Request failed with status ${response.status}.`);
    }

    return text;
}

async function startCamera() {
    try {
        if (stream) {
            stopCamera();
        }

        liveSessionId = createSessionId();
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = stream;
        cameraPlaceholder.classList.add("hidden");
        setLiveStatus("Starting webcam analysis...");

        video.onloadedmetadata = () => {
            overlay.width = video.videoWidth;
            overlay.height = video.videoHeight;
            video.play();

            clearInterval(liveInterval);
            liveInterval = setInterval(sendFrame, 1200);
            sendFrame();
            setLiveStatus("Camera is running. Keep your face inside the frame.");
        };
    } catch (error) {
        console.error("Camera error:", error);
        setLiveStatus("Camera access failed. Check browser permission and backend status.", true);
        cameraPlaceholder.classList.remove("hidden");
    }
}

function stopCamera() {
    clearInterval(liveInterval);
    liveInterval = null;
    liveRequestInFlight = false;

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    video.pause();
    video.srcObject = null;
    cameraPlaceholder.classList.remove("hidden");
    overlayContext.clearRect(0, 0, overlay.width, overlay.height);
    resetLiveMetrics();
    setLiveStatus("Camera is idle.");
}

async function sendFrame() {
    if (!stream || liveRequestInFlight || video.videoWidth === 0 || video.readyState < 2) {
        return;
    }

    liveRequestInFlight = true;

    try {
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext("2d");
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.9));
        if (!blob) {
            throw new Error("Could not capture the current video frame.");
        }

        const formData = new FormData();
        formData.append("session_id", liveSessionId);
        formData.append("frame", blob, "frame.jpg");

        const data = await fetchJson(`${API}/analyze-frame`, {
            method: "POST",
            body: formData
        });

        if (!data.success) {
            throw new Error("Frame analysis did not complete.");
        }

        if (!stream) {
            return;
        }

        if (!data.face_detected) {
            renderNoFaceState();
            setLiveStatus("No face detected. Move closer or improve lighting.");
            return;
        }

        updateLiveMetrics(data);
        drawFaceBox(data.face_box, data.verdict);
        updateLiveBarChart(data.emotions);
        setLiveStatus("Live analysis is updating normally.");
    } catch (error) {
        console.error("Frame API error:", error);
        setLiveStatus(error.message || "Live analysis request failed.", true);
    } finally {
        liveRequestInFlight = false;
    }
}

function drawFaceBox(box, verdict) {
    overlayContext.clearRect(0, 0, overlay.width, overlay.height);
    if (!box) {
        return;
    }

    let color = "#10b981";
    if (verdict === "deceptive") {
        color = "#ef4444";
    } else if (verdict === "uncertain") {
        color = "#eab308";
    }

    overlayContext.strokeStyle = color;
    overlayContext.lineWidth = 4;
    overlayContext.strokeRect(box.x, box.y, box.w, box.h);

    const label = formatLabel(verdict).toUpperCase();
    overlayContext.font = "600 16px sans-serif";
    const labelWidth = overlayContext.measureText(label).width + 16;
    const labelX = box.x;
    const labelY = Math.max(box.y - 28, 6);

    overlayContext.fillStyle = color;
    overlayContext.fillRect(labelX, labelY, labelWidth, 24);
    overlayContext.fillStyle = "#ffffff";
    overlayContext.fillText(label, labelX + 8, labelY + 17);
}

function setUploadBusy(isBusy) {
    const button = document.getElementById("uploadBtn");
    button.disabled = isBusy;
    button.textContent = isBusy ? "Analyzing..." : "Analyze Video File";
}

async function uploadVideo() {
    const file = document.getElementById("videoFile").files[0];
    if (!file) {
        setUploadError("Select a video file first.");
        return;
    }

    setUploadError("");
    setUploadBusy(true);
    setUploadStatus(`Analyzing ${file.name}... this can take a few seconds.`);

    try {
        const formData = new FormData();
        formData.append("video", file);

        const data = await fetchJson(`${API}/analyze`, {
            method: "POST",
            body: formData
        });

        if (!data.success) {
            throw new Error("Video analysis did not complete.");
        }

        displayVideoResults(data);
        updateTimelineChart(data.timeline || []);
        setUploadStatus(`Analysis completed for ${file.name}.`);
    } catch (error) {
        console.error("Upload error:", error);
        setUploadError(error.message || "Upload failed.");
        setUploadStatus("Video analysis failed.", true);
    } finally {
        setUploadBusy(false);
    }
}

function displayVideoResults(data) {
    const resultsPanel = document.getElementById("videoResultsPanel");
    resultsPanel.classList.remove("hidden");

    const riskScore = document.getElementById("riskScoreVal");
    const riskLevel = document.getElementById("riskLevelVal");
    const dominant = document.getElementById("videoDominantVal");
    const faces = document.getElementById("videoFacesVal");

    riskScore.textContent = String(data.risk.score);
    riskScore.style.color = data.risk.color;
    riskLevel.textContent = data.risk.level;
    riskLevel.style.color = data.risk.color;
    dominant.textContent = formatLabel(data.summary.dominant_emotion);
    faces.textContent = String(data.summary.faces_detected);

    document.getElementById("videoDurationVal").textContent = `Duration: ${formatDuration(data.video_info.duration)}`;
    document.getElementById("videoFramesVal").textContent = `Samples: ${data.video_info.frames_analyzed}`;
    document.getElementById("videoPeakVal").textContent = `Peak Score: ${data.summary.peak_deception}`;

    const list = document.getElementById("indicatorsList");
    list.innerHTML = "";

    if (!data.risk.indicators.length) {
        const item = document.createElement("li");
        item.className = "indicator-item muted";
        item.textContent = "No strong deception indicators were detected in the uploaded clip.";
        list.appendChild(item);
        return;
    }

    data.risk.indicators.forEach(indicator => {
        const item = document.createElement("li");
        item.className = "indicator-item";

        if (indicator.severity === "high") {
            item.style.borderColor = "#ef4444";
        } else if (indicator.severity === "medium") {
            item.style.borderColor = "#eab308";
        } else {
            item.style.borderColor = "#3b82f6";
        }

        item.innerHTML = `<strong>${indicator.name}</strong><span>${indicator.description}</span>`;
        list.appendChild(item);
    });
}

function updateLiveBarChart(emotions) {
    const labels = Object.keys(emotions);
    const values = Object.values(emotions);
    const colors = labels.map(label => emotionColors[label] || "#cbd5e1");

    if (liveChartInstance) {
        liveChartInstance.data.labels = labels;
        liveChartInstance.data.datasets[0].data = values;
        liveChartInstance.data.datasets[0].backgroundColor = colors;
        liveChartInstance.update();
        return;
    }

    const chartContext = document.getElementById("liveEmotionChart").getContext("2d");
    liveChartInstance = new Chart(chartContext, {
        type: "bar",
        data: {
            labels,
            datasets: [
                {
                    label: "Emotion %",
                    data: values,
                    backgroundColor: colors,
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 250 },
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: "#334155" },
                    ticks: { color: "#cbd5e1" }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: "#cbd5e1" }
                }
            }
        }
    });
}

function updateTimelineChart(timeline) {
    const panel = document.getElementById("timelinePanel");
    if (!timeline.length) {
        panel.classList.add("hidden");
        if (timelineChartInstance) {
            timelineChartInstance.destroy();
            timelineChartInstance = null;
        }
        return;
    }

    panel.classList.remove("hidden");

    const labels = timeline.map(point => `${point.timestamp.toFixed(1)}s`);
    const negativeSignal = timeline.map(point => {
        const emotions = point.emotions || EMPTY_EMOTIONS;
        return Number((emotions.fear + emotions.angry + emotions.disgust).toFixed(2));
    });
    const deceptionSignal = timeline.map(point => point.deception_score || 0);
    const neutralSignal = timeline.map(point => (point.emotions ? point.emotions.neutral : 0));

    if (timelineChartInstance) {
        timelineChartInstance.destroy();
    }

    const chartContext = document.getElementById("videoTimelineChart").getContext("2d");
    timelineChartInstance = new Chart(chartContext, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: "Deception Score",
                    data: deceptionSignal,
                    borderColor: "#38bdf8",
                    backgroundColor: "rgba(56, 189, 248, 0.12)",
                    fill: true,
                    tension: 0.3,
                    borderWidth: 2
                },
                {
                    label: "Negative Emotion Signal",
                    data: negativeSignal,
                    borderColor: "#ef4444",
                    backgroundColor: "rgba(239, 68, 68, 0.10)",
                    fill: true,
                    tension: 0.28,
                    borderWidth: 2
                },
                {
                    label: "Neutral %",
                    data: neutralSignal,
                    borderColor: "#94a3b8",
                    tension: 0.22,
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: "index",
                intersect: false
            },
            plugins: {
                legend: {
                    position: "top",
                    labels: { color: "#cbd5e1" }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: "#334155" },
                    ticks: { color: "#cbd5e1" }
                },
                x: {
                    grid: { color: "#1e293b" },
                    ticks: {
                        color: "#cbd5e1",
                        maxTicksLimit: 10
                    }
                }
            }
        }
    });
}

function formatDuration(durationSeconds) {
    if (!durationSeconds || durationSeconds <= 0) {
        return "0s";
    }

    const totalSeconds = Math.round(durationSeconds);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;

    if (minutes === 0) {
        return `${seconds}s`;
    }

    return `${minutes}m ${seconds}s`;
}

window.addEventListener("beforeunload", stopCamera);
resetLiveMetrics();
