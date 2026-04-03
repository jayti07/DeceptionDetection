import os
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from analyzer import EMOTIONS, analyze_frame_emotion, analyze_video, calculate_deception_risk

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "webm", "mkv"}
MAX_UPLOAD_SIZE_BYTES = 150 * 1024 * 1024
LIVE_HISTORY_LIMIT = 12
LIVE_SESSION_TTL_SECONDS = 5 * 60
LIVE_SESSION_LIMIT = 100
LIVE_SESSIONS = {}

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_BYTES


def ensure_upload_folder():
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def calculate_live_deception(emotions, dominant, history):
    fear = emotions.get("fear", 0.0)
    angry = emotions.get("angry", 0.0)
    disgust = emotions.get("disgust", 0.0)
    sad = emotions.get("sad", 0.0)
    surprise = emotions.get("surprise", 0.0)
    neutral = emotions.get("neutral", 0.0)
    happy = emotions.get("happy", 0.0)

    negative_pressure = (
        fear * 0.35
        + angry * 0.32
        + disgust * 0.18
        + sad * 0.10
        + surprise * 0.05
    )

    guarded_neutrality = 0.0
    if neutral > 28:
        guarded_neutrality = max(0.0, negative_pressure - 10.0) * 0.55
        guarded_neutrality += max(0.0, neutral - 45.0) * 0.35

    mixed_signal = 0.0
    if surprise > 16 and negative_pressure > 18:
        mixed_signal += 6.0
    if happy > 12 and negative_pressure > 22:
        mixed_signal += 5.0

    volatility = 0.0
    if history:
        recent_negative = [
            frame["emotions"].get("fear", 0.0) * 0.35
            + frame["emotions"].get("angry", 0.0) * 0.32
            + frame["emotions"].get("disgust", 0.0) * 0.18
            + frame["emotions"].get("sad", 0.0) * 0.10
            + frame["emotions"].get("surprise", 0.0) * 0.05
            for frame in history
        ]
        average_negative = sum(recent_negative) / len(recent_negative)
        spike = negative_pressure - average_negative

        if history[-1]["dominant"] != dominant:
            volatility += 10.0
        if spike > 4:
            volatility += min(18.0, spike * 1.6)

        dominant_switches = sum(
            1
            for index in range(1, len(history))
            if history[index]["dominant"] != history[index - 1]["dominant"]
        )
        if len(history) >= 4 and dominant_switches >= max(1, len(history) // 3):
            volatility += 8.0

    score = negative_pressure * 0.85 + guarded_neutrality + mixed_signal + volatility

    if dominant in {"fear", "angry", "disgust"}:
        score += 8.0
    elif dominant == "sad":
        score += 4.0

    if dominant == "happy" and negative_pressure < 12:
        score *= 0.65
    if dominant == "neutral" and negative_pressure < 10:
        score *= 0.55

    return int(round(max(0.0, min(score, 100.0))))


def determine_verdict(score):
    if score >= 60:
        return "deceptive"
    if score >= 35:
        return "uncertain"
    return "truthful"


def cleanup_live_sessions():
    now = time.time()
    expired_session_ids = [
        session_id
        for session_id, session in LIVE_SESSIONS.items()
        if now - session["updated_at"] > LIVE_SESSION_TTL_SECONDS
    ]
    for session_id in expired_session_ids:
        LIVE_SESSIONS.pop(session_id, None)


def get_live_history(session_id):
    cleanup_live_sessions()

    if session_id not in LIVE_SESSIONS:
        if len(LIVE_SESSIONS) >= LIVE_SESSION_LIMIT:
            oldest_session_id = min(
                LIVE_SESSIONS,
                key=lambda current_session_id: LIVE_SESSIONS[current_session_id]["updated_at"],
            )
            LIVE_SESSIONS.pop(oldest_session_id, None)

        LIVE_SESSIONS[session_id] = {
            "history": deque(maxlen=LIVE_HISTORY_LIMIT),
            "updated_at": time.time(),
        }

    LIVE_SESSIONS[session_id]["updated_at"] = time.time()
    return LIVE_SESSIONS[session_id]["history"]


def enrich_timeline(frames):
    history = deque(maxlen=LIVE_HISTORY_LIMIT)
    enriched_frames = []

    for frame in frames:
        frame_data = dict(frame)

        if frame_data["face_detected"]:
            frame_score = calculate_live_deception(
                frame_data["emotions"],
                frame_data["dominant"],
                list(history),
            )
            verdict = determine_verdict(frame_score)
            history.append(
                {
                    "dominant": frame_data["dominant"],
                    "emotions": frame_data["emotions"],
                }
            )
        else:
            frame_score = 0
            verdict = "no-face"

        frame_data["deception_score"] = frame_score
        frame_data["verdict"] = verdict
        enriched_frames.append(frame_data)

    return enriched_frames


def build_video_summary(frames):
    detected_frames = [frame for frame in frames if frame["face_detected"]]
    if not detected_frames:
        return {
            "faces_detected": 0,
            "dominant_emotion": "none",
            "peak_deception": 0,
            "average_emotions": {emotion: 0.0 for emotion in EMOTIONS},
        }

    average_emotions = {
        emotion: round(
            float(np.mean([frame["emotions"].get(emotion, 0.0) for frame in detected_frames])),
            2,
        )
        for emotion in EMOTIONS
    }

    dominant_emotion = max(average_emotions, key=average_emotions.get)
    peak_deception = max(frame.get("deception_score", 0) for frame in detected_frames)

    return {
        "faces_detected": len(detected_frames),
        "dominant_emotion": dominant_emotion,
        "peak_deception": int(peak_deception),
        "average_emotions": average_emotions,
    }


@app.errorhandler(413)
def file_too_large(_error):
    max_megabytes = int(MAX_UPLOAD_SIZE_BYTES / (1024 * 1024))
    return jsonify({"error": f"Video is too large. Maximum supported size is {max_megabytes} MB."}), 413


@app.route("/")
def home():
    return jsonify({"status": "DeceptIQ API running"})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():
    ensure_upload_folder()

    if "video" not in request.files:
        return jsonify({"error": "No video file provided."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({"error": f"Invalid file type. Supported formats: {allowed}."}), 400

    extension = file.filename.rsplit(".", 1)[1].lower()
    filepath = UPLOAD_FOLDER / f"{uuid.uuid4()}.{extension}"
    file.save(str(filepath))

    try:
        video_data = analyze_video(str(filepath), target_fps=2.0, max_samples=180)
        if "error" in video_data:
            return jsonify({"error": video_data["error"]}), 500

        timeline = enrich_timeline(video_data["frames"])
        risk_data = calculate_deception_risk(timeline)
        summary = build_video_summary(timeline)

        return jsonify(
            {
                "success": True,
                "video_info": {
                    "duration": video_data["duration"],
                    "frames_analyzed": video_data["total_analyzed"],
                    "fps": video_data["fps"],
                    "sample_interval_frames": video_data["sample_interval_frames"],
                },
                "summary": summary,
                "risk": risk_data,
                "timeline": timeline,
            }
        )
    except Exception as exc:
        print(f"Video analysis error: {exc}")
        return jsonify({"error": f"Video analysis failed: {exc}"}), 500
    finally:
        if filepath.exists():
            filepath.unlink()


@app.route("/analyze-frame", methods=["POST"])
def analyze_frame():
    if "frame" not in request.files:
        return jsonify({"error": "No frame provided."}), 400

    session_id = (request.form.get("session_id") or "default").strip() or "default"
    file = request.files["frame"]
    img_bytes = file.read()
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Could not decode uploaded frame."}), 400

    try:
        result = analyze_frame_emotion(frame)
        if result is None:
            return jsonify(
                {
                    "success": True,
                    "face_detected": False,
                    "emotions": {emotion: 0.0 for emotion in EMOTIONS},
                    "dominant": "none",
                    "deception_score": 0,
                    "verdict": "unknown",
                    "face_box": None,
                }
            )

        history = get_live_history(session_id)
        emotions = result["emotions"]
        dominant = result["dominant"]
        deception_score = calculate_live_deception(emotions, dominant, list(history))
        verdict = determine_verdict(deception_score)

        history.append({"dominant": dominant, "emotions": emotions})

        return jsonify(
            {
                "success": True,
                "face_detected": True,
                "emotions": emotions,
                "dominant": dominant,
                "deception_score": deception_score,
                "verdict": verdict,
                "face_box": result["face_box"],
            }
        )
    except Exception as exc:
        print(f"Frame analysis error: {exc}")
        return jsonify({"error": f"Frame analysis failed: {exc}"}), 500


@app.route("/analyze-demo", methods=["GET"])
def analyze_demo():
    import random

    frames = []
    history = deque(maxlen=LIVE_HISTORY_LIMIT)

    for index in range(20):
        spike_frame = index in {8, 14}
        emotions = {
            "neutral": round(15.0 if spike_frame else random.uniform(38, 62), 2),
            "happy": round(random.uniform(4, 15), 2),
            "sad": round(random.uniform(3, 12), 2),
            "angry": round(42.0 if spike_frame else random.uniform(4, 11), 2),
            "fear": round(28.0 if spike_frame else random.uniform(3, 10), 2),
            "disgust": round(16.0 if spike_frame else random.uniform(2, 7), 2),
            "surprise": round(random.uniform(2, 8), 2),
        }

        total = sum(emotions.values())
        normalized = {
            emotion: round((value / total) * 100.0, 2)
            for emotion, value in emotions.items()
        }
        dominant = "angry" if spike_frame else "neutral"
        frame_score = calculate_live_deception(normalized, dominant, list(history))
        history.append({"dominant": dominant, "emotions": normalized})

        frames.append(
            {
                "frame": index * 5,
                "timestamp": round(index * 0.5, 2),
                "emotions": normalized,
                "dominant": dominant,
                "face_detected": True,
                "deception_score": frame_score,
                "verdict": determine_verdict(frame_score),
            }
        )

    risk = calculate_deception_risk(frames)
    summary = build_video_summary(frames)

    return jsonify(
        {
            "success": True,
            "video_info": {"duration": 10.0, "frames_analyzed": 20, "fps": 30.0},
            "summary": summary,
            "risk": risk,
            "timeline": frames,
        }
    )


if __name__ == "__main__":
    ensure_upload_folder()
    print("DeceptIQ backend starting on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
