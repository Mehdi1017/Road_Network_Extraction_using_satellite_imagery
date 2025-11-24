import os
import glob
import numpy as np
from PIL import Image

# UPDATE THIS to match your exact folder structure
DATA_ROOT = '/home/mehdi/thesis/AOI_2_Vegas'
MASK_FOLDER = os.path.join(DATA_ROOT, 'PS-RGB-Masks')

print(f"--- INSPECTING: {MASK_FOLDER} ---")

# 1. Check if the folder exists
if not os.path.exists(MASK_FOLDER):
    print(f"❌ ERROR: Folder does not exist!")
    exit()

# 2. Check if files exist inside
files = glob.glob(os.path.join(MASK_FOLDER, '*.png'))
print(f"✅ Found {len(files)} .png files.")

if len(files) == 0:
    print("❌ ERROR: Folder exists but contains no PNG files.")
    exit()

# 3. Check the content of the first 5 masks
print("\n--- CHECKING CONTENT OF FIRST 5 MASKS ---")
count_empty = 0
count_valid = 0

for i, fpath in enumerate(files[:5]):
    filename = os.path.basename(fpath)
    try:
        # Load image
        mask = np.array(Image.open(fpath))
        unique_vals = np.unique(mask)
        
        # Check if it has roads
        has_roads = (1 in unique_vals) or (255 in unique_vals)
        
        status = "✅ HAS ROADS" if has_roads else "⚠️ ALL BLACK (Empty)"
        print(f"[{i}] {filename} | Values: {unique_vals} | {status}")
        
        if not has_roads:
            count_empty += 1
        else:
            count_valid += 1
            
    except Exception as e:
        print(f"❌ Could not open {filename}: {e}")

# 4. Debug Path Logic (Simulating dataset.py)
print("\n--- DEBUGGING PATH MATCHING LOGIC ---")
# Get a real image path to test against
sample_img_path = os.path.join(DATA_ROOT, 'PS-RGB', filename.replace('.png', '.tif'))
print(f"Mock Image Path: {sample_img_path}")

# This is the exact logic from your dataset.py:
calculated_mask_path = sample_img_path.replace('PS-RGB', 'PS-RGB-Masks').replace('.tif', '.png')

print(f"Calculated Path: {calculated_mask_path}")
print(f"Real Path:       {os.path.join(MASK_FOLDER, filename)}")

if calculated_mask_path == os.path.join(MASK_FOLDER, filename):
    print("✅ Path logic MATCHES.")
else:
    print("❌ Path logic FAILS. See difference above.")