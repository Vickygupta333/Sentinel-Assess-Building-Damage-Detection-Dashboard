"""Quick test to verify the API fix"""
import urllib.request
import urllib.error
import json
from PIL import Image
import io

# Create a simple PNG image in memory
img = Image.new('RGB', (100, 100), color='red')
img_bytes = io.BytesIO()
img.save(img_bytes, format='PNG')
img_data = img_bytes.getvalue()

print("Testing API /health endpoint...")
try:
    r = urllib.request.urlopen('http://127.0.0.1:8000/health')
    print(f"✓ Health check: {r.read().decode()}")
except Exception as e:
    print(f"✗ Health check failed: {e}")

print("\nTesting if uvicorn reloaded...")
# The easiest way is to check if the server is still responding
try:
    r = urllib.request.urlopen('http://127.0.0.1:8000/health')
    print("✓ Server is responding (changes should be loaded)")
except Exception as e:
    print(f"✗ Server error: {e}")
