from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
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

# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def validate_image_format(filename: str) -> bool:
    """Validate if the uploaded file has a supported image format."""
    return Path(filename).suffix.lower() in SUPPORTED_FORMATS

def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded file to a temporary location."""
    try:
        suffix = Path(upload_file.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = upload_file.file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
            logger.info(f"File saved to {tmp_file_path}")
            return tmp_file_path
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

def cleanup_temp_file(file_path: str):
    """Remove temporary file safely."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Temporary file {file_path} removed")
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary file {file_path}: {str(e)}")

def detect_emotion(image_path: str):
    """Detect emotion from image using DeepFace and return emotion + level."""
    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=True,
            detector_backend='mtcnn'
        )
        if isinstance(result, list):
            emotions = result[0]['emotion']
        else:
            emotions = result['emotion']
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]
        # Normalize confidence to 1-10 scale
        level = int(round(confidence / 10))
        level = max(1, min(level, 10))  # Ensure level is between 1 and 10
        logger.info(f"Detected emotion: {dominant_emotion} with confidence: {confidence:.2f} (level: {level})")
        return dominant_emotion, level
    except ValueError as e:
        if "Face could not be detected" in str(e):
            logger.warning("No face detected in the image")
            raise HTTPException(status_code=400, detail="No face detected in the image")
        else:
            logger.error(f"ValueError during emotion detection: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during emotion detection: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal error during emotion analysis")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Face Emotion Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "analyze_emotion": "/analyze-emotion (POST)"
        }
    }

@app.post("/analyze-emotion")
async def analyze_emotion(file: UploadFile = File(...)):
    """
    Analyze emotion from uploaded image.
    """
    temp_file_path = None
    try:
        if not validate_image_format(file.filename):
            logger.warning(f"Unsupported image format: {Path(file.filename).suffix}")
            raise HTTPException(status_code=400, detail="Unsupported image format. Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp")
        temp_file_path = save_uploaded_file(file)
        emotion, level = detect_emotion(temp_file_path)
        return JSONResponse(
            status_code=200,
            content={"emotion": emotion, "level": level}
        )
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if temp_file_path:
            cleanup_temp_file(temp_file_path)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Face Emotion Detection API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
