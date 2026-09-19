"""Test the API endpoint to diagnose the 400 error"""
from fastapi.testclient import TestClient
from main import app
import io
from PIL import Image
import json

client = TestClient(app)

# Create test images
def make_test_image():
    img = Image.new('RGB', (200, 200), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes

pre_file = make_test_image()
post_file = make_test_image()

print("Testing /api/infer endpoint...")
print("=" * 50)
response = client.post(
    "/api/infer",
    files={
        "pre_image": ("pre.png", pre_file, "image/png"),
        "post_image": ("post.png", post_file, "image/png"),
    }
)

print(f"Status Code: {response.status_code}")
print(f"Response Headers: {dict(response.headers)}")
print(f"Response Body: {response.text[:500]}")

if response.status_code == 200:
    print("\n✓ SUCCESS! Response JSON:")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"\n✗ Error ({response.status_code}): {response.text}")
