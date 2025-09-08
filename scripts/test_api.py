"""
Test script for the Face Emotion Detection API.
Run this after starting the FastAPI server to test the endpoints.
"""

import requests
import json
from pathlib import Path

# API base URL (adjust if running on different host/port)
BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health check endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print("✅ Health endpoint working\n")
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}\n")

def test_root_endpoint():
    """Test the root endpoint."""
    print("Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print("✅ Root endpoint working\n")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}\n")

def test_emotion_analysis_no_file():
    """Test emotion analysis endpoint without file."""
    print("Testing emotion analysis without file...")
    try:
        response = requests.post(f"{BASE_URL}/analyze-emotion")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        print("✅ Error handling working (no file)\n")
    except Exception as e:
        print(f"❌ No file test failed: {e}\n")

def test_emotion_analysis_with_sample():
    """Test emotion analysis with a sample image (if available)."""
    print("Testing emotion analysis with sample image...")
    print("Note: This requires a sample image file named 'sample_face.jpg' in the same directory")
    
    sample_file = Path("sample_face.jpg")
    if not sample_file.exists():
        print("⚠️ No sample image found. Create a 'sample_face.jpg' file to test this endpoint\n")
        return
    
    try:
        with open(sample_file, "rb") as f:
            files = {"file": ("sample_face.jpg", f, "image/jpeg")}
            response = requests.post(f"{BASE_URL}/analyze-emotion", files=files)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Emotion analysis working\n")
        else:
            print("⚠️ Emotion analysis returned error (check image has visible face)\n")
    except Exception as e:
        print(f"❌ Emotion analysis test failed: {e}\n")

if __name__ == "__main__":
    print("🧪 Testing Face Emotion Detection API\n")
    print("Make sure the FastAPI server is running on http://localhost:8000\n")
    
    test_health_endpoint()
    test_root_endpoint()
    test_emotion_analysis_no_file()
    test_emotion_analysis_with_sample()
    
    print("🏁 Testing complete!")
    print("\n📖 To test manually, visit: http://localhost:8000/docs")
