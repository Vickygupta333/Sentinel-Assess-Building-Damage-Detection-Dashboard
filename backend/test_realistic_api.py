"""Test with realistic building-like imagery"""
import urllib.request
import urllib.error
import json
import io
import numpy as np
from PIL import Image

print("="*60)
print("Testing with Realistic Building Imagery")
print("="*60)

# Create synthetic satellite-like images
def create_satellite_image(damaged=False, seed=42):
    """Create a realistic satellite image with buildings"""
    np.random.seed(seed)
    
    # Create ground texture (greenish/brownish)
    img = np.random.randint(40, 80, size=(400, 400, 3), dtype=np.uint8)
    
    # Add some grass/vegetation variation
    for i in range(400):
        for j in range(400):
            img[i, j, 1] += 20  # boost green channel for grass-like look
    
    # Draw buildings (rectangular structures with different colors)
    buildings = [
        (50, 50, 150, 150, [150, 140, 130]),      # tan/beige building
        (200, 80, 320, 180, [160, 150, 140]),      # another tan building
        (80, 250, 180, 330, [140, 150, 160]),      # grayish building
        (250, 240, 380, 350, [130, 125, 120]),     # dark building
    ]
    
    for x1, y1, x2, y2, color in buildings:
        img[y1:y2, x1:x2] = color
        
        # Add damage to one building in the post image
        if damaged and (x1, y1) == (200, 80):
            # Add rubble/damage texture
            damage = np.random.randint(80, 150, size=(y2-y1, x2-x1, 3), dtype=np.uint8)
            img[y1:y2, x1:x2] = damage
    
    return Image.fromarray(img.astype(np.uint8))

# Create pre and post images
pre_img = create_satellite_image(damaged=False)
post_img = create_satellite_image(damaged=True)

print("\n1. Created realistic satellite images:")
print("   - Pre image: 400x400 with 4 buildings")
print("   - Post image: 400x400 with 1 building showing damage")

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

body.write(f'--{boundary}\r\n'.encode())
body.write(b'Content-Disposition: form-data; name="pre_image"; filename="pre.png"\r\n')
body.write(b'Content-Type: image/png\r\n\r\n')
body.write(pre_bytes.getvalue())
body.write(b'\r\n')

body.write(f'--{boundary}\r\n'.encode())
body.write(b'Content-Disposition: form-data; name="post_image"; filename="post.png"\r\n')
body.write(b'Content-Type: image/png\r\n\r\n')
body.write(post_bytes.getvalue())
body.write(b'\r\n')

body.write(f'--{boundary}--\r\n'.encode())

print(f"\n2. Sending request to /api/infer...")

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
    print(f"   Waiting for response (analyzing images)...\n")
    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.loads(response.read().decode())
        
        print(f"✓ SUCCESS! API Response received\n")
        print(f"3. Analysis Results:")
        print(f"   └─ Image size: {result['image_size']} pixels")
        print(f"   └─ Model: {result['model_mode']}")
        print(f"   └─ Total buildings: {result['summary']['total_buildings']}")
        print(f"   └─ Flagged as damaged: {result['summary']['damaged']}")
        print(f"   └─ Undamaged: {result['summary']['undamaged']}")
        
        if result['buildings']:
            print(f"\n4. Detected Buildings:")
            for b in sorted(result['buildings'], key=lambda x: x['damage_score'], reverse=True):
                status = "⚠ DAMAGED" if b['damaged'] else "✓ OK"
                print(f"   [{b['id']}] Score: {b['damage_score']:.2f} - {status}")
                print(f"        Bounding box: {b['bbox']}")
        else:
            print(f"\n4. No buildings detected")
            print(f"   (The building detector might need tuning for this image)")

        print("\n✓ API and backend are working!")

except urllib.error.HTTPError as e:
    error_response = e.read().decode()
    print(f"✗ HTTP {e.code} Error!")
    print(f"   Response: {error_response[:200]}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
