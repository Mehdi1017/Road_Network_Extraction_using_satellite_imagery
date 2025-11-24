# preprocess.py
import os
import glob
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from tqdm import tqdm

DATA_DIR = './AOI_2_Vegas'
MASK_DIR = os.path.join(DATA_DIR, 'PS-RGB-Masks')
os.makedirs(MASK_DIR, exist_ok=True)

# Find all the label files
print(os.path.join(DATA_DIR, 'geojson_roads'))
label_files = glob.glob(os.path.join(DATA_DIR, 'geojson_roads' , '*.geojson'))

print(f"Found {len(label_files)} label files. Starting preprocessing...")

for label_path in tqdm(label_files):
    try:
        # Find matching image file to get metadata
        base_name = os.path.basename(label_path).replace('geojson_roads', 'PS-RGB').replace('.geojson', '.tif')
        image_path = os.path.join(DATA_DIR, 'PS-RGB', base_name)
        
        with rasterio.open(image_path) as src:
            meta = src.meta
        
        # Create an empty mask
        mask = np.zeros((meta['height'], meta['width']), dtype=np.uint8)
        
        # Load labels and create mask
        labels_gdf = gpd.read_file(label_path)
        if not labels_gdf.empty:
                    # A. Save the original CRS (likely EPSG:4326)
            original_crs = labels_gdf.crs

            # B. Project to a metric CRS (EPSG:3857 is 'Web Mercator', used by Google Maps)
            # This allows us to calculate "2 meters" accurately.
            labels_proj = labels_gdf.to_crs(epsg=3857)

            # C. Perform the buffer (2 meters)
            # This creates the polygon shape of the road
            labels_proj['geometry'] = labels_proj.geometry.buffer(2)

            # D. Project BACK to the original CRS
            # We must do this so the polygons align with your satellite image again
            labels_gdf = labels_proj.to_crs(original_crs)
            shapes_to_burn = [(geom, 255) for geom in labels_gdf.geometry]
            mask = rasterize(
                shapes=shapes_to_burn,
                out_shape=(meta['height'], meta['width']),
                transform=meta['transform'],
                fill=0,
                dtype=np.uint8
            )
        
        # Save the mask as a simple PNG
        mask_save_path = os.path.join(MASK_DIR, base_name.replace('.tif', '.png'))
        
        # Use rasterio to save as a geo-referenced file (or use cv2.imwrite for simple PNG)
        meta.update(driver='PNG', count=1, dtype='uint8', nodata=None)
        with rasterio.open(mask_save_path, 'w', **meta) as dst:
            dst.write(mask, 1)
            
    except Exception as e:
        print(f"Error processing {label_path}: {e}")

print("Preprocessing complete.")