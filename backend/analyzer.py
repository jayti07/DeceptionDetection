import cv2
import numpy as np

# Load OpenCV's built-in face detector (no extra downloads needed)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def analyze_frame_emotion(frame):
    """
    Estimates emotion using facial geometry + brightness analysis.
    This is a simplified heuristic model (no heavy AI library needed).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    # Take the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_gray = gray[y:y+h, x:x+w]
    face_color = frame[y:y+h, x:x+w]

    # Resize to standard size for analysis
    face_gray = cv2.resize(face_gray, (64, 64))
    face_color = cv2.resize(face_color, (64, 64))

    # --- Feature Extraction ---

    # 1. Overall brightness (low = sad/fear, high = happy)
    brightness = np.mean(face_gray) / 255.0

    # 2. Upper face region (forehead/brows) vs lower (mouth)
    upper = face_gray[:32, :]
    lower = face_gray[32:, :]
    upper_mean = np.mean(upper) / 255.0
    lower_mean = np.mean(lower) / 255.0

    # 3. Edge intensity — high edges = expressive face
    edges = cv2.Canny(face_gray, 50, 150)
    edge_intensity = np.sum(edges) / (64 * 64 * 255)

    # 4. Contrast (std deviation) — high contrast = strong expression
    contrast = np.std(face_gray) / 128.0

    # 5. Color channel analysis
    b, g, r = cv2.split(face_color)
    r_mean = np.mean(r) / 255.0
    g_mean = np.mean(g) / 255.0

    # --- Emotion Scoring (heuristic rules) ---
    scores = {}

    # Happy: bright, high lower face activity, higher red
    scores['happy'] = (brightness * 0.4 + lower_mean * 0.3 + r_mean * 0.2 + edge_intensity * 0.1)

    # Neutral: medium everything, low edges
    scores['neutral'] = (1 - abs(brightness - 0.5)) * 0.5 + (1 - edge_intensity) * 0.3 + (1 - contrast) * 0.2

    # Sad: low brightness, low edge activity
    scores['sad'] = ((1 - brightness) * 0.5 + (1 - edge_intensity) * 0.3 + (upper_mean - lower_mean + 0.5) * 0.2)

    # Angry: high contrast, high edges, upper face tension
    scores['angry'] = (contrast * 0.4 + edge_intensity * 0.3 + upper_mean * 0.2 + (1 - brightness) * 0.1)

    # Fear: low brightness, high edges, uneven face
    scores['fear'] = ((1 - brightness) * 0.4 + edge_intensity * 0.3 + abs(upper_mean - lower_mean) * 0.3)

    # Disgust: asymmetry + contrast
    scores['disgust'] = (contrast * 0.5 + abs(r_mean - g_mean) * 0.3 + edge_intensity * 0.2)

    # Surprise: very high edges, high brightness
    scores['surprise'] = (edge_intensity * 0.5 + brightness * 0.3 + contrast * 0.2)

    # Normalize to percentages
    total = sum(scores.values())
    if total == 0:
        emotions_pct = {e: round(100/7, 2) for e in EMOTIONS}
    else:
        emotions_pct = {k: round((v / total) * 100, 2) for k, v in scores.items()}

    dominant = max(emotions_pct, key=emotions_pct.get)

    return emotions_pct, dominant


def analyze_video(video_path, sample_rate=5):
    """
    Analyzes a video file for micro-expressions.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Cannot open video file"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    results = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_rate == 0:
            timestamp = frame_count / fps if fps > 0 else 0

            result = analyze_frame_emotion(frame)

            if result:
                emotions_pct, dominant = result
                results.append({
                    'frame': frame_count,
                    'timestamp': round(timestamp, 2),
                    'emotions': emotions_pct,
                    'dominant': dominant,
                    'face_detected': True
                })
            else:
                results.append({
                    'frame': frame_count,
                    'timestamp': round(timestamp, 2),
                    'emotions': {e: 0.0 for e in EMOTIONS},
                    'dominant': 'none',
                    'face_detected': False
                })

        frame_count += 1

    cap.release()

    return {
        'frames': results,
        'fps': fps,
        'duration': round(duration, 2),
        'total_analyzed': len(results)
    }


def calculate_deception_risk(frame_results):
    """
    Calculates Deception Risk Score (0-100)
    """
    frames = [f for f in frame_results if f['face_detected']]

    if len(frames) < 3:
        return {
            'score': 0,
            'level': 'Insufficient Data',
            'indicators': [],
            'color': 'gray'
        }

    indicators = []
    score = 0

    # --- INDICATOR 1: Emotional Volatility ---
    dominant_emotions = [f['dominant'] for f in frames]
    changes = sum(1 for i in range(1, len(dominant_emotions))
                  if dominant_emotions[i] != dominant_emotions[i-1])
    volatility = changes / len(dominant_emotions)

    if volatility > 0.5:
        score += 35
        indicators.append({
            'name': 'High Emotional Volatility',
            'description': f'Dominant emotion changed {changes} times across {len(frames)} samples',
            'severity': 'high'
        })
    elif volatility > 0.3:
        score += 15
        indicators.append({
            'name': 'Moderate Emotional Volatility',
            'description': 'Some inconsistency in emotional expression detected',
            'severity': 'medium'
        })

    # --- INDICATOR 2: Micro-Expression Spikes ---
    negative_spikes = 0
    for i in range(1, len(frames) - 1):
        prev = frames[i-1]['emotions']
        curr = frames[i]['emotions']
        for emotion in ['fear', 'disgust', 'angry']:
            if curr.get(emotion, 0) - prev.get(emotion, 0) > 20:
                negative_spikes += 1

    if negative_spikes > 2:
        score += 30
        indicators.append({
            'name': 'Micro-Expression Spikes Detected',
            'description': f'{negative_spikes} sudden negative emotion bursts found',
            'severity': 'high'
        })
    elif negative_spikes > 0:
        score += 10
        indicators.append({
            'name': 'Minor Emotional Spikes',
            'description': f'{negative_spikes} brief emotional spike(s) detected',
            'severity': 'low'
        })

    # --- INDICATOR 3: Suppression Pattern ---
    neutral_frames = [f for f in frames if f['dominant'] == 'neutral']
    neutral_ratio = len(neutral_frames) / len(frames)
    avg_negative = np.mean([
        (f['emotions'].get('fear', 0) +
         f['emotions'].get('disgust', 0) +
         f['emotions'].get('angry', 0)) / 3
        for f in frames
    ])

    if neutral_ratio > 0.6 and avg_negative > 15:
        score += 25
        indicators.append({
            'name': 'Emotional Suppression Pattern',
            'description': 'Predominantly neutral face but underlying negative traces detected',
            'severity': 'medium'
        })

    score = min(score, 100)

    if score >= 70:
        level, color = 'High Risk', '#ef4444'
    elif score >= 40:
        level, color = 'Moderate Risk', '#f59e0b'
    elif score >= 15:
        level, color = 'Low Risk', '#3b82f6'
    else:
        level, color = 'Minimal Risk', '#10b981'

    return {
        'score': score,
        'level': level,
        'indicators': indicators,
        'color': color
    }