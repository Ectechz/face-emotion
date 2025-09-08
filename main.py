from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from deepface import DeepFace
from PIL import Image
import logging
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face Emotion Detection API",
    description="API for detecting emotions from facial images using DeepFace",
    version="1.0.0"
)

# Serve static files for UI (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def validate_image_format(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_FORMATS

def save_uploaded_file(upload_file: UploadFile) -> str:
    suffix = Path(upload_file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(upload_file.file.read())
        return tmp_file.name

def cleanup_temp_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)

def detect_emotion(image_path: str):
    result = DeepFace.analyze(
        img_path=image_path,
        actions=['emotion'],
        enforce_detection=True,
        detector_backend='mtcnn'
    )
    emotions = result[0]['emotion'] if isinstance(result, list) else result['emotion']
    dominant_emotion = max(emotions, key=emotions.get)
    confidence = emotions[dominant_emotion]
    level = max(1, min(int(round(confidence / 10)), 10))
    return dominant_emotion, level

# --- UI Route ---
@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Emotion Detection</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: auto; padding: 20px; }
            h1 { text-align: center; }
            input[type=file] { display: block; margin: 20px 0; }
            button { padding: 10px 20px; }
            #result { margin-top: 20px; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Face Emotion Detection</h1>
        <form id="uploadForm">
            <input type="file" id="fileInput" name="file" accept="image/*" required>
            <button type="submit">Detect Emotion</button>
        </form>
        <div id="result"></div>
        <script>
            const form = document.getElementById('uploadForm');
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                const formData = new FormData();
                formData.append('file', file);
                const resultDiv = document.getElementById('result');
                resultDiv.textContent = 'Analyzing...';
                try {
                    const response = await fetch('/analyze-emotion', { method: 'POST', body: formData });
                    const data = await response.json();
                    resultDiv.textContent = `Detected Emotion: ${data.emotion}, Level: ${data.level}`;
                } catch (err) {
                    resultDiv.textContent = 'Error detecting emotion.';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# --- API Route ---
@app.post("/analyze-emotion")
async def analyze_emotion(file: UploadFile = File(...)):
    temp_file_path = None
    try:
        if not validate_image_format(file.filename):
            raise HTTPException(status_code=400, detail="Unsupported image format.")
        temp_file_path = save_uploaded_file(file)
        emotion, level = detect_emotion(temp_file_path)
        return JSONResponse({"emotion": emotion, "level": level})
    finally:
        if temp_file_path:
            cleanup_temp_file(temp_file_path)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Face Emotion Detection API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
