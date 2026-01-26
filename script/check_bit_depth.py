import rasterio
import numpy as np
import os

# --- CONFIGURATION ---
# Point this to one of your .tif images
IMAGE_PATH = "/home/mehdi/thesis/src/AOI_2_Vegas/PS-RGB/SN3_roads_train_AOI_2_Vegas_PS-RGB_img126.tif"

def check_image_stats():
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ File not found: {IMAGE_PATH}")
        return

    try:
        with rasterio.open(IMAGE_PATH) as src:
            print(f"--- Image Metadata ---")
            print(f"Filename: {os.path.basename(IMAGE_PATH)}")
            print(f"Driver: {src.driver}")
            print(f"Width: {src.width}, Height: {src.height}")
            print(f"Count (Bands): {src.count}")
            print(f"Data Type (dtype): {src.dtypes[0]}") # Likely 'uint16'
            
            # Read the data to find max values
            data = src.read()
            max_val = np.max(data)
            min_val = np.min(data)
            
            print(f"\n--- Pixel Value Statistics ---")
            print(f"Min Value: {min_val}")
            print(f"Max Value: {max_val}")
            
            print(f"\n--- Analysis ---")
            if src.dtypes[0] == 'uint16':
                print("Container: 16-bit (0-65535)")
                if max_val <= 2047:
                    print("Conclusion: Data fits within 11-bit range (0-2047). Confirmed SpaceNet 11-bit data.")
                else:
                    print(f"Conclusion: Data exceeds 11-bit range ({max_val} > 2047). Possibly pre-processed or 16-bit.")
            elif src.dtypes[0] == 'uint8':
                print("Container: 8-bit (0-255). This image has already been downsampled.")

    except Exception as e:
        print(f"Error reading image: {e}")

if __name__ == "__main__":
    check_image_stats()