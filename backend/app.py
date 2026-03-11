from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import cv2
import numpy as np
from analyzer import analyze_video, calculate_deception_risk, analyze_frame_emotion

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return jsonify({'status': 'DeceptIQ API v2.0 running!'})


# ── VIDEO UPLOAD ANALYSIS ────────────────────
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        print(f"Analyzing video: {filename}")
        video_data = analyze_video(filepath, sample_rate=5)

        if 'error' in video_data:
            return jsonify({'error': video_data['error']}), 500

        risk_data = calculate_deception_risk(video_data['frames'])

        return jsonify({
            'success': True,
            'video_info': {
                'duration': video_data['duration'],
                'frames_analyzed': video_data['total_analyzed'],
                'fps': video_data['fps']
            },
            'risk': risk_data,
            'timeline': video_data['frames']
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# ── LIVE FRAME ANALYSIS (WEBCAM) ─────────────
@app.route('/analyze-frame', methods=['POST'])
def analyze_frame():
    """
    Receives a single JPEG frame from the webcam
    Returns emotion analysis + deception verdict
    """
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400

    file = request.files['frame']
    
    # Read image from upload
    img_bytes = file.read()
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Could not decode image'}), 400

    try:
        result = analyze_frame_emotion(frame)

        if result is None:
            return jsonify({
                'success': True,
                'face_detected': False,
                'emotions': {},
                'dominant': 'none',
                'deception_score': 0,
                'verdict': 'unknown'
            })

        emotions_pct, dominant = result

        # Calculate instant deception score
        deception_score = calculate_instant_deception(emotions_pct, dominant)

        # Determine verdict
        if deception_score >= 65:
            verdict = 'deceptive'
        elif deception_score >= 35:
            verdict = 'uncertain'
        else:
            verdict = 'truthful'

        # Get face box for overlay drawing
        face_box = get_face_box(frame)

        return jsonify({
            'success': True,
            'face_detected': True,
            'emotions': emotions_pct,
            'dominant': dominant,
            'deception_score': deception_score,
            'verdict': verdict,
            'face_box': face_box
        })

    except Exception as e:
        print(f"Frame analysis error: {e}")
        return jsonify({'error': str(e)}), 500


def calculate_instant_deception(emotions, dominant):
    """
    Calculates a real-time deception score for a single frame
    """
    score = 0

    # High fear + suppressed neutral = deception signal
    fear = emotions.get('fear', 0)
    angry = emotions.get('angry', 0)
    disgust = emotions.get('disgust', 0)
    neutral = emotions.get('neutral', 0)

    negative_sum = fear + angry + disgust

    # Strong negative emotions hidden behind neutral face
    if neutral > 40 and negative_sum > 30:
        score += 40

    # Strong individual negative emotion
    if fear > 35:
        score += 25
    if angry > 35:
        score += 20
    if disgust > 30:
        score += 15

    # Multiple negative emotions at once = higher suspicion
    if fear > 15 and angry > 15:
        score += 20

    return min(score, 100)


def get_face_box(frame):
    """Returns face bounding box coordinates"""
    import cv2
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}


# ── DEMO ENDPOINT ────────────────────────────
@app.route('/analyze-demo', methods=['GET'])
def analyze_demo():
    import random

    frames = []
    for i in range(20):
        base_neutral = random.uniform(40, 70)
        spike_frame = i == 8 or i == 14

        frames.append({
            'frame': i * 5,
            'timestamp': round(i * 0.5, 2),
            'emotions': {
                'neutral':  10.0 if spike_frame else base_neutral,
                'happy':    random.uniform(5, 20),
                'sad':      random.uniform(2, 10),
                'angry':    45.0 if spike_frame else random.uniform(1, 8),
                'fear':     30.0 if spike_frame else random.uniform(1, 6),
                'disgust':  random.uniform(1, 5),
                'surprise': random.uniform(1, 8),
            },
            'dominant': 'angry' if spike_frame else 'neutral',
            'face_detected': True
        })

    risk = calculate_deception_risk(frames)

    return jsonify({
        'success': True,
        'video_info': {'duration': 10.0, 'frames_analyzed': 20, 'fps': 30.0},
        'risk': risk,
        'timeline': frames
    })


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    print("🚀 DeceptIQ Backend v2.0 Starting...")
    print("📡 API: http://localhost:5000")
    print("📹 Live webcam endpoint: /analyze-frame")
    app.run(debug=True, port=5000)