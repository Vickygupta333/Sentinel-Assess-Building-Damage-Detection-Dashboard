"""Test the /api/infer endpoint with a real request"""
import io
from PIL import Image
import urllib.request
import urllib.parse
import json

# Create a simple test image
img = Image.new('RGB', (200, 200), color='red')
img_bytes = io.BytesIO()
img.save(img_bytes, format='PNG')
img_bytes.seek(0)

# Test 1: Check endpoint directly with curl-like request
print("Testing API /api/infer endpoint...")
print("=" * 50)

# Create a simple test with urllib
url = 'http://127.0.0.1:8000/api/infer'

# Create a boundary for multipart
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'

# Prepare the multipart body
body = []
body.append(f'--{boundary}'.encode())
body.append(b'Content-Disposition: form-data; name="pre_image"; filename="pre.png"')
body.append(b'Content-Type: image/png')
body.append(b'')
img.save(img_bytes := io.BytesIO(), format='PNG')
body.append(img_bytes.getvalue())
body.append(f'--{boundary}'.encode())
body.append(b'Content-Disposition: form-data; name="post_image"; filename="post.png"')
body.append(b'Content-Type: image/png')
body.append(b'')
img.save(img_bytes := io.BytesIO(), format='PNG')
body.append(img_bytes.getvalue())
body.append(f'--{boundary}--'.encode())

body_bytes = b'\r\n'.join(body)

# Send request
req = urllib.request.Request(
    url,
    data=body_bytes,
    headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}'
    },
    method='POST'
)

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())
        print("✓ SUCCESS! API Response:")
        print(json.dumps(result, indent=2))
except urllib.error.HTTPError as e:
    print(f"✗ HTTP {e.code} Error")
    print(f"Response: {e.read().decode()}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
