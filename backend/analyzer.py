import cv2
import numpy as np

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
SMILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def detect_primary_face(gray_frame):
    if gray_frame is None or gray_frame.size == 0:
        return None

    faces = FACE_CASCADE.detectMultiScale(
        gray_frame,
        scaleFactor=1.08,
        minNeighbors=6,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        faces = FACE_CASCADE.detectMultiScale(
            gray_frame,
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=(40, 40),
        )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    return int(x), int(y), int(w), int(h)


def _clip_metric(value, lower=0.0, upper=1.0):
    return float(np.clip(value, lower, upper))


def _edge_density(region, threshold_low=40, threshold_high=120):
    if region is None or region.size == 0:
        return 0.0

    edges = cv2.Canny(region, threshold_low, threshold_high)
    return _clip_metric(np.mean(edges) / 255.0)


def _normalized_std(region, divisor=64.0):
    if region is None or region.size == 0:
        return 0.0

    return _clip_metric(np.std(region) / divisor)


def _eye_presence_score(upper_face):
    if upper_face is None or upper_face.size == 0:
        return 0.0

    eyes = EYE_CASCADE.detectMultiScale(
        upper_face,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(12, 12),
    )
    return _clip_metric(min(len(eyes), 2) / 2.0)


def _smile_score(lower_face):
    if lower_face is None or lower_face.size == 0:
        return 0.0

    smiles = SMILE_CASCADE.detectMultiScale(
        lower_face,
        scaleFactor=1.7,
        minNeighbors=20,
        minSize=(18, 18),
    )
    if len(smiles) == 0:
        return 0.0

    smile_width = max(smile[2] for smile in smiles)
    return _clip_metric(smile_width / max(lower_face.shape[1], 1))


def _prepare_face(face_gray, face_color):
    face_gray = cv2.resize(face_gray, (96, 96), interpolation=cv2.INTER_LINEAR)
    face_color = cv2.resize(face_color, (96, 96), interpolation=cv2.INTER_LINEAR)
    face_gray = CLAHE.apply(face_gray)
    face_color = cv2.GaussianBlur(face_color, (3, 3), 0)
    return face_gray, face_color


def _pick_dominant_emotion(emotions_pct):
    ranked = sorted(emotions_pct.items(), key=lambda item: item[1], reverse=True)
    leader, runner_up = ranked[0], ranked[1]

    if leader[0] in {"sad", "fear", "angry", "disgust"}:
        if emotions_pct["neutral"] >= leader[1] - 2.5:
            return "neutral"

    if leader[1] - runner_up[1] < 3.0 and emotions_pct["neutral"] >= runner_up[1]:
        return "neutral"

    return leader[0]


def _score_emotions(face_gray, face_color):
    upper = face_gray[:36, :]
    lower = face_gray[56:, :]
    mouth = face_gray[58:88, 18:78]
    left = face_gray[:, :48]
    right = np.fliplr(face_gray[:, 48:])

    brightness = _clip_metric(np.mean(face_gray) / 255.0)
    contrast = _normalized_std(face_gray)
    upper_brightness = _clip_metric(np.mean(upper) / 255.0)
    lower_brightness = _clip_metric(np.mean(lower) / 255.0)
    upper_edge = _edge_density(upper)
    lower_edge = _edge_density(lower)
    mouth_edge = _edge_density(mouth)
    eye_presence = _eye_presence_score(upper)
    smile_strength = _smile_score(lower)
    symmetry_gap = _clip_metric(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))) / 90.0)
    upper_lower_gap = _clip_metric(upper_brightness - lower_brightness + 0.35)
    edge_balance = _clip_metric(1.0 - abs(upper_edge - lower_edge))
    mouth_openness = _clip_metric(mouth_edge * 1.6 + lower_edge * 0.5)

    hsv = cv2.cvtColor(face_color, cv2.COLOR_BGR2HSV)
    saturation = _clip_metric(np.mean(hsv[:, :, 1]) / 255.0)
    blue_mean, green_mean, red_mean = [np.mean(channel) / 255.0 for channel in cv2.split(face_color)]
    warmth = _clip_metric(red_mean - blue_mean + 0.3)
    color_balance = _clip_metric(1.0 - abs(red_mean - green_mean))

    scores = {
        "neutral": 0.28
        + (1.0 - abs(brightness - 0.55)) * 0.22
        + (1.0 - mouth_openness) * 0.12
        + (1.0 - symmetry_gap) * 0.10
        + eye_presence * 0.08
        + edge_balance * 0.08,
        "happy": 0.08
        + smile_strength * 0.34
        + lower_edge * 0.18
        + brightness * 0.15
        + warmth * 0.12
        + mouth_openness * 0.08
        + saturation * 0.05,
        "sad": 0.06
        + (1.0 - brightness) * 0.20
        + upper_lower_gap * 0.20
        + (1.0 - smile_strength) * 0.10
        + (1.0 - lower_edge) * 0.10
        + (1.0 - saturation) * 0.06,
        "angry": 0.05
        + upper_edge * 0.24
        + contrast * 0.20
        + symmetry_gap * 0.16
        + (1.0 - brightness) * 0.10
        + eye_presence * 0.08,
        "fear": 0.05
        + eye_presence * 0.20
        + mouth_openness * 0.20
        + contrast * 0.14
        + symmetry_gap * 0.10
        + abs(upper_brightness - lower_brightness) * 0.10
        + (1.0 - brightness) * 0.06,
        "disgust": 0.04
        + symmetry_gap * 0.26
        + contrast * 0.18
        + upper_lower_gap * 0.16
        + (1.0 - warmth) * 0.08
        + upper_edge * 0.06,
        "surprise": 0.05
        + eye_presence * 0.22
        + mouth_openness * 0.24
        + brightness * 0.14
        + contrast * 0.10
        + lower_edge * 0.08
        + color_balance * 0.04,
    }

    if smile_strength > 0.18:
        scores["happy"] += 0.18
        scores["sad"] *= 0.82

    if mouth_openness > 0.34 and eye_presence > 0.4:
        scores["surprise"] += 0.16
        scores["fear"] += 0.07

    if symmetry_gap > 0.3 and upper_edge > 0.22:
        scores["angry"] += 0.10
        scores["disgust"] += 0.05

    powered_scores = {emotion: max(score, 0.01) ** 1.3 for emotion, score in scores.items()}
    total = sum(powered_scores.values())
    emotions_pct = {
        emotion: round((value / total) * 100.0, 2)
        for emotion, value in powered_scores.items()
    }
    return emotions_pct


def analyze_frame_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    face_box = detect_primary_face(gray)
    if face_box is None:
        return None

    x, y, w, h = face_box
    pad_x = max(int(w * 0.08), 4)
    pad_y = max(int(h * 0.12), 6)
    x0 = max(x - pad_x, 0)
    y0 = max(y - pad_y, 0)
    x1 = min(x + w + pad_x, frame.shape[1])
    y1 = min(y + h + pad_y, frame.shape[0])

    face_gray = gray[y0:y1, x0:x1]
    face_color = frame[y0:y1, x0:x1]
    prepared_gray, prepared_color = _prepare_face(face_gray, face_color)

    emotions_pct = _score_emotions(prepared_gray, prepared_color)
    dominant = _pick_dominant_emotion(emotions_pct)

    return {
        "emotions": emotions_pct,
        "dominant": dominant,
        "face_box": {
            "x": int(x0),
            "y": int(y0),
            "w": int(x1 - x0),
            "h": int(y1 - y0),
        },
    }


def _build_frame_result(frame_number, timestamp, analysis):
    if analysis is None:
        return {
            "frame": int(frame_number),
            "timestamp": round(float(timestamp), 2),
            "emotions": {emotion: 0.0 for emotion in EMOTIONS},
            "dominant": "none",
            "face_detected": False,
        }

    return {
        "frame": int(frame_number),
        "timestamp": round(float(timestamp), 2),
        "emotions": analysis["emotions"],
        "dominant": analysis["dominant"],
        "face_detected": True,
    }


def analyze_video(video_path, target_fps=2.0, max_samples=180):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Cannot open video file"}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = round(total_frames / fps, 2) if fps > 0 and total_frames > 0 else 0.0
    sample_interval_frames = max(int(round(fps / target_fps)), 1) if fps > 0 else 10

    results = []

    if total_frames > 0:
        sample_positions = np.arange(0, total_frames, sample_interval_frames, dtype=int)
        if len(sample_positions) > max_samples:
            sample_positions = np.linspace(
                0,
                max(total_frames - 1, 0),
                num=max_samples,
                dtype=int,
            )

        for frame_number in np.unique(sample_positions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ret, frame = cap.read()
            if not ret:
                continue

            timestamp = frame_number / fps if fps > 0 else 0.0
            analysis = analyze_frame_emotion(frame)
            results.append(_build_frame_result(frame_number, timestamp, analysis))

    if not results:
        cap.release()
        cap = cv2.VideoCapture(video_path)
        frame_number = 0

        while len(results) < max_samples:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % sample_interval_frames == 0:
                timestamp = frame_number / fps if fps > 0 else 0.0
                analysis = analyze_frame_emotion(frame)
                results.append(_build_frame_result(frame_number, timestamp, analysis))

            frame_number += 1

    cap.release()

    return {
        "frames": results,
        "fps": round(fps, 2) if fps > 0 else 0.0,
        "duration": duration,
        "total_analyzed": len(results),
        "sample_interval_frames": int(sample_interval_frames),
    }


def calculate_deception_risk(frame_results):
    frames = [frame for frame in frame_results if frame["face_detected"]]
    if len(frames) < 3:
        return {
            "score": 0,
            "level": "Insufficient Data",
            "indicators": [],
            "color": "gray",
        }

    indicators = []
    score = 0

    dominant_emotions = [frame["dominant"] for frame in frames]
    changes = sum(
        1
        for index in range(1, len(dominant_emotions))
        if dominant_emotions[index] != dominant_emotions[index - 1]
    )
    volatility = changes / max(len(dominant_emotions) - 1, 1)

    if volatility > 0.45:
        score += 30
        indicators.append(
            {
                "name": "High Emotional Volatility",
                "description": f"Dominant emotion changed {changes} times across {len(frames)} sampled moments.",
                "severity": "high",
            }
        )
    elif volatility > 0.25:
        score += 15
        indicators.append(
            {
                "name": "Moderate Emotional Volatility",
                "description": "Facial expression shifted multiple times during the clip.",
                "severity": "medium",
            }
        )

    negative_spikes = 0
    for index in range(1, len(frames)):
        previous = frames[index - 1]["emotions"]
        current = frames[index]["emotions"]
        for emotion in ("fear", "disgust", "angry"):
            if current.get(emotion, 0.0) - previous.get(emotion, 0.0) > 12:
                negative_spikes += 1

    if negative_spikes >= 4:
        score += 25
        indicators.append(
            {
                "name": "Repeated Negative Emotion Spikes",
                "description": f"{negative_spikes} sudden rises in fear, disgust, or anger were detected.",
                "severity": "high",
            }
        )
    elif negative_spikes > 0:
        score += 10
        indicators.append(
            {
                "name": "Brief Negative Emotion Spikes",
                "description": f"{negative_spikes} short-lived negative spikes were detected.",
                "severity": "low",
            }
        )

    neutral_ratio = sum(1 for frame in frames if frame["dominant"] == "neutral") / len(frames)
    avg_negative = float(
        np.mean(
            [
                frame["emotions"].get("fear", 0.0)
                + frame["emotions"].get("disgust", 0.0)
                + frame["emotions"].get("angry", 0.0)
                for frame in frames
            ]
        )
    )

    if neutral_ratio > 0.45 and avg_negative > 35:
        score += 20
        indicators.append(
            {
                "name": "Suppression Pattern",
                "description": "Neutral expression remained common while negative cues stayed elevated.",
                "severity": "medium",
            }
        )

    frame_scores = [frame.get("deception_score", 0) for frame in frames]
    average_frame_score = float(np.mean(frame_scores))
    peak_frame_score = max(frame_scores)

    if average_frame_score >= 55:
        score += 20
        indicators.append(
            {
                "name": "Sustained Elevated Deception Signal",
                "description": f"Average frame score stayed high at {average_frame_score:.1f}.",
                "severity": "high",
            }
        )
    elif average_frame_score >= 35:
        score += 10
        indicators.append(
            {
                "name": "Moderately Elevated Deception Signal",
                "description": f"Average frame score reached {average_frame_score:.1f}.",
                "severity": "medium",
            }
        )

    if peak_frame_score >= 75:
        score += 10
        indicators.append(
            {
                "name": "Peak Stress Moment",
                "description": f"At least one sampled frame reached a score of {peak_frame_score}.",
                "severity": "medium",
            }
        )

    score = min(int(round(score)), 100)

    if score >= 70:
        level, color = "High Risk", "#ef4444"
    elif score >= 40:
        level, color = "Moderate Risk", "#f59e0b"
    elif score >= 15:
        level, color = "Low Risk", "#3b82f6"
    else:
        level, color = "Minimal Risk", "#10b981"

    return {
        "score": score,
        "level": level,
        "indicators": indicators,
        "color": color,
    }
