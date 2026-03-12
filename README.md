# DeceptIQ — Truth Detection System

An AI-powered web application that analyzes facial micro-expressions
to detect hidden emotions and calculate a Deception Risk Score in real-time.

## Features
- 🎥 Live webcam analysis with instant Truthful / Uncertain / Deceptive verdict
- 📁 Video file upload with full emotion timeline report
- 📊 Tracks 7 emotions — happy, sad, angry, fear, disgust, surprise, neutral
- 🚩 Detects 3 deception indicators — volatility, micro-spikes, suppression
- 📈 Interactive charts — emotion timeline and distribution
- ⚡ Deception Risk Score from 0 to 100

## Tech Stack
- **Frontend:** HTML, CSS, JavaScript, Chart.js
- **Backend:** Python, Flask, OpenCV
- **Communication:** REST API

## Run Locally
```bash
cd backend
venv\Scripts\activate
python app.py
```
Open `frontend/index.html` with Live Server.

## Disclaimer
Research-based indicator only — not a clinical lie detector.
Emotional inconsistency ≠ guaranteed deception.
