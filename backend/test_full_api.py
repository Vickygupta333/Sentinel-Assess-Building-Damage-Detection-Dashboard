"""Test the complete API flow with image upload"""
import urllib.request
import urllib.error
import json
import io
from PIL import Image

print("="*60)
print("Testing Complete API Pipeline")
print("="*60)

# Create synthetic test images
def create_test_image(size=200, color='red'):
    img = Image.new('RGB', (size, size), color=color)
    return img

# Create pre and post images
pre_img = create_test_image(200, 'red')
post_img = create_test_image(200, 'red')

# Add some "damage" to post image
pixels_post = post_img.load()
for i in range(50, 100):
    for j in range(50, 100):
        pixels_post[i, j] = (100, 50, 50)  # darker red = "damage"

# Convert to bytes
pre_bytes = io.BytesIO()
pre_img.save(pre_bytes, format='PNG')
pre_bytes.seek(0)

post_bytes = io.BytesIO()
post_img.save(post_bytes, format='PNG')
post_bytes.seek(0)

# Create multipart/form-data request
boundary = '----FormBoundary7MA4YWxkTrZu0gW'
body = io.BytesIO()

# Add pre_image part
body.write(f'--{boundary}\r\n'.encode())
body.write(b'Content-Disposition: form-data; name="pre_image"; filename="test_pre.png"\r\n')
body.write(b'Content-Type: image/png\r\n\r\n')
body.write(pre_bytes.getvalue())
body.write(b'\r\n')

# Add post_image part
body.write(f'--{boundary}\r\n'.encode())
body.write(b'Content-Disposition: form-data; name="post_image"; filename="test_post.png"\r\n')
body.write(b'Content-Type: image/png\r\n\r\n')
body.write(post_bytes.getvalue())
body.write(b'\r\n')

# Close boundary
body.write(f'--{boundary}--\r\n'.encode())

print(f"\n1. Created test images: pre.png and post.png")
print(f"   Pre image: 200x200 red")
print(f"   Post image: 200x200 red with damage (darker patch)")

print(f"\n2. Sending request to /api/infer...")
print(f"   Content-Type: multipart/form-data; boundary={boundary}")
print(f"   Body size: {body.tell()} bytes")

# Send the request
url = 'http://127.0.0.1:8000/api/infer'
req = urllib.request.Request(
    url,
    data=body.getvalue(),
    headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}'
    },
    method='POST'
)

try:
    print(f"\n3. Waiting for response...")
    with urllib.request.urlopen(req, timeout=30) as response:
        response_data = response.read()
        result = json.loads(response_data.decode())
        
        print(f"\n✓ SUCCESS! Status: {response.status}")
        print(f"\n4. Response Analysis:")
        print(f"   - Image size: {result['image_size']}")
        print(f"   - Model mode: {result['model_mode']}")
        print(f"   - Total buildings detected: {result['summary']['total_buildings']}")
        print(f"   - Damaged: {result['summary']['damaged']}")
        print(f"   - Undamaged: {result['summary']['undamaged']}")
        
        if result['buildings']:
            print(f"\n5. Building Details:")
            for b in result['buildings']:
                print(f"   - {b['id']}: score={b['damage_score']:.3f}, damaged={b['damaged']}")
        else:
            print(f"\n   No buildings detected (this is normal for solid-color images)")
        
        print(f"\n✓ API Pipeline is working correctly!")
        
except urllib.error.HTTPError as e:
    print(f"\n✗ HTTP {e.code} Error!")
    error_response = e.read().decode()
    print(f"   Error: {error_response}")
    print(f"\n   Status line: {e.msg}")
    
except urllib.error.URLError as e:
    print(f"\n✗ Connection Error: {e.reason}")
    print(f"   Is the server running on http://127.0.0.1:8000?")
    
except json.JSONDecodeError as e:
    print(f"\n✗ JSON Parsing Error: {e}")
    print(f"   Response was not valid JSON")
    
except Exception as e:
    print(f"\n✗ Unexpected Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
