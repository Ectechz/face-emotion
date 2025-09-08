# Face Emotion Detection API

A FastAPI application that detects emotions from facial images using the DeepFace library.

## Features

- Upload image via POST endpoint
- Detect faces in uploaded images
- Classify emotions using pretrained models
- Clean JSON response with detected emotion
- Comprehensive error handling
- Swagger UI documentation
- Health check endpoint

## Installation

1. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

2. Run the application:
\`\`\`bash
python main.py
\`\`\`

Or using uvicorn directly:
\`\`\`bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
\`\`\`

## Usage

### API Endpoints

- `GET /` - Root endpoint with API information
- `POST /analyze-emotion` - Upload image and get emotion analysis
- `GET /health` - Health check endpoint
- `GET /docs` - Swagger UI documentation

### Example Usage

\`\`\`python
import requests

# Upload an image for emotion analysis
with open("your_image.jpg", "rb") as f:
    files = {"file": ("image.jpg", f, "image/jpeg")}
    response = requests.post("http://localhost:8000/analyze-emotion", files=files)
    
print(response.json())  # {"emotion": "happy"}
\`\`\`

### Supported Image Formats

- JPG/JPEG
- PNG
- BMP
- TIFF
- WebP

### Supported Emotions

The API can detect the following emotions:
- happy
- sad
- angry
- fear
- surprise
- disgust
- neutral

## Error Handling

The API handles various error cases:
- No file uploaded
- Invalid file format
- File too large (>10MB)
- No face detected in image
- Invalid image file
- Internal processing errors

## Testing

Run the test script to verify all endpoints:
\`\`\`bash
python scripts/test_api.py
\`\`\`

## API Documentation

Once the server is running, visit `http://localhost:8000/docs` for interactive Swagger UI documentation.
