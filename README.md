# DeceptIQ

DeceptIQ is a browser-based facial-expression analysis project with:

- live webcam emotion analysis
- rolling deception scoring for live frames
- uploaded video analysis with timeline charts and key indicators

## Project Structure

- `backend/` - Flask API and OpenCV analysis logic
- `frontend/` - static HTML, CSS, and JavaScript dashboard

## Setup

Create and activate a virtual environment, then install the Python dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run The Backend

```bash
cd backend
python app.py
```

The API starts on `http://127.0.0.1:5000`.

## Open The Frontend

Serve the `frontend/` folder with a local static server such as VS Code Live Server, then open:

```text
frontend/index.html
```

The frontend calls the Flask backend on port `5000`.

## Supported Video Formats

- `.mp4`
- `.avi`
- `.mov`
- `.webm`
- `.mkv`

## Notes

- The webcam flow uses rolling session history so the score reacts to changes over time instead of staying stuck at zero.
- Uploaded videos are sampled across the clip to keep analysis responsive and to avoid missing later moments.
- This is a heuristic research demo, not a forensic lie detector.
