"""
Simulate exactly what the browser frontend does
"""
import urllib.request
import urllib.error
import json
import io
from PIL import Image

print("="*70)
print("BROWSER-LIKE TEST: Simulating Frontend Upload")
print("="*70)

# Create test images exactly like the frontend would
def create_test_images():
    """Create two test images with detectable buildings"""
    pre = Image.new('RGB', (300, 300), color=(100, 120, 100))  # greenish ground
    post = Image.new('RGB', (300, 300), color=(100, 120, 100))  # greenish ground
    
    # Add buildings to pre image
    pre_pix = pre.load()
    post_pix = post.load()
    
    for x in range(80, 150):
        for y in range(80, 150):
            pre_pix[x, y] = (140, 130, 120)  # tan building
            post_pix[x, y] = (140, 130, 120)  # same building in post
    
    for x in range(200, 280):
        for y in range(100, 200):
            pre_pix[x, y] = (120, 110, 100)  # gray building
            post_pix[x, y] = (80, 70, 60)    # damaged building in post
    
    return pre, post

# Create and save images
print("\n1. Creating test images...")
pre_img, post_img = create_test_images()
print("   ✓ Pre-image: 300x300 with 2 buildings")
print("   ✓ Post-image: 300x300 with 1 intact and 1 damaged building")

# Convert to bytes (like browser FileReader would)
print("\n2. Converting images to bytes (like browser would)...")
pre_bytes = io.BytesIO()
pre_img.save(pre_bytes, format='PNG')
pre_data = pre_bytes.getvalue()

post_bytes = io.BytesIO()
post_img.save(post_bytes, format='PNG')
post_data = post_bytes.getvalue()

print(f"   ✓ Pre-image: {len(pre_data)} bytes")
print(f"   ✓ Post-image: {len(post_data)} bytes")

# Create FormData-like multipart request (exactly like browser)
print("\n3. Creating multipart/form-data request...")
boundary = '----WebKitFormBoundary' + 'A' * 16
body = io.BytesIO()

# Add first file (pre_image)
body.write(f'--{boundary}\r\n'.encode())
body.write(b'Content-Disposition: form-data; name="pre_image"; filename="test_pre.png"\r\n')
body.write(b'Content-Type: image/png\r\n')
body.write(b'\r\n')
body.write(pre_data)
body.write(b'\r\n')

# Add second file (post_image)
body.write(f'--{boundary}\r\n'.encode())
body.write(b'Content-Disposition: form-data; name="post_image"; filename="test_post.png"\r\n')
body.write(b'Content-Type: image/png\r\n')
body.write(b'\r\n')
body.write(post_data)
body.write(b'\r\n')

# End boundary
body.write(f'--{boundary}--\r\n'.encode())

body_data = body.getvalue()
print(f"   ✓ Total request size: {len(body_data)} bytes")

# Send request to backend
print("\n4. Sending request to http://127.0.0.1:8000/api/infer...")
url = 'http://127.0.0.1:8000/api/infer'

req = urllib.request.Request(
    url,
    data=body_data,
    headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    },
    method='POST'
)

try:
    print("   Waiting for response...")
    with urllib.request.urlopen(req, timeout=30) as response:
        response_data = response.read()
        result = json.loads(response_data.decode())
        
        print(f"\n✓✓✓ SUCCESS! ✓✓✓")
        print(f"\n5. Response received (Status: {response.status}):")
        print(f"   └─ Image size: {result['image_size']} pixels")
        print(f"   └─ Model: {result['model_mode']}")
        print(f"   └─ Total buildings: {result['summary']['total_buildings']}")
        print(f"   └─ Damaged: {result['summary']['damaged']}")
        print(f"   └─ Undamaged: {result['summary']['undamaged']}")
        
        if result['buildings']:
            print(f"\n6. Buildings detected:")
            for b in sorted(result['buildings'], key=lambda x: x['damage_score'], reverse=True):
                print(f"   [{b['id']}] Score: {b['damage_score']:.3f}")
                print(f"       Damaged: {b['damaged']}")
                print(f"       BBox: {b['bbox']}")
        
        print(f"\n✓ FRONTEND CAN UPLOAD AND PROCESS IMAGES SUCCESSFULLY!")
        print(f"\nIf you're not seeing results in the browser:")
        print(f"  1. Open DevTools (F12) → Console")
        print(f"  2. Check for JavaScript errors")
        print(f"  3. Verify API_BASE is set correctly")
        print(f"  4. Make sure the server is running: http://127.0.0.1:8000")
        
except urllib.error.HTTPError as e:
    print(f"\n✗ HTTP ERROR {e.code}")
    error_text = e.read().decode()
    print(f"   Error response: {error_text[:500]}")
    
except urllib.error.URLError as e:
    print(f"\n✗ CONNECTION ERROR: {e.reason}")
    print(f"   Is the backend running on http://127.0.0.1:8000?")
    
except json.JSONDecodeError:
    print(f"\n✗ JSON PARSING ERROR")
    print(f"   Response was not valid JSON")
    print(f"   Response text: {response_data.decode()[:200]}")
    
except Exception as e:
    print(f"\n✗ UNEXPECTED ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
