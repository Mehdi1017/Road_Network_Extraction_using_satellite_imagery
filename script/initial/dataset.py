import os
import glob
import numpy as np
import torch
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from torch.utils.data import Dataset
from PIL import Image

class RoadDataset(Dataset):
    def __init__(self, file_list_path=None, data_root=None, transform=None):
        self.transform = transform
        
        # --- 1. DEFINE PATHS (Guaranteed to run) ---
        # We hardcode these to match your verified folder structure
        self.mask_dir = '../src/AOI_2_Vegas/PS-RGB-Masks'
        self.geojson_dir = '../src/AOI_2_Vegas/geojson_roads'
        
        # --- 2. LOAD IMAGE LIST ---
        if file_list_path:
            # Training/Test mode: Read from text file
            with open(file_list_path, 'r') as f:
                self.image_files = [line.strip() for line in f.readlines()]
                
        elif data_root:
            # Quick Test mode: Scan folder
            self.image_files = glob.glob(os.path.join(data_root, 'PS-RGB', 'src', '*.tif'))
            
        else:
            raise ValueError("Must provide either file_list_path or data_root")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. Load Image
        image_path = self.image_files[idx]
        with rasterio.open(image_path) as src:
            image = src.read() # (C, H, W)
            meta = src.meta    # Keep metadata for on-the-fly rasterizing

        # 2. LOAD MASK (Hybrid Strategy)
        # Strategy A: Look for pre-processed PNG
        base_name_png = os.path.basename(image_path).replace('.tif', '.png')
        mask_path_png = os.path.join(self.mask_dir, base_name_png)
        
        mask = None

        # Check if PNG exists
        if os.path.exists(mask_path_png):
            try:
                mask = np.array(Image.open(mask_path_png))
            except:
                pass 

        # Strategy B: Generate from GeoJSON if PNG failed
        if mask is None:
            # Construct GeoJSON path
            base_name_geojson = os.path.basename(image_path).replace('PS-RGB', 'geojson_roads').replace('.tif', '.geojson')
            geojson_path = os.path.join(self.geojson_dir, base_name_geojson)
            
            if os.path.exists(geojson_path):
                mask = self.create_mask_from_geojson(geojson_path, meta)
            else:
                # Strategy C: No label exists -> Empty Black Mask
                mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)

        # 3. Fixed Normalization (Safer for Transformers)
        # SpaceNet 3 is 11-bit data (0-2047). 
        # We simply divide by 2048 to map it to 0-1.
        # This preserves the natural "darkness" of shadows/desert 
        # and prevents noise amplification.
        image = image.astype(np.float32) / 2048.0
            
        # Cast to float32 for OpenCV/Albumentations compatibility
        image = image.astype(np.float32)
            
        # 4. Augmentation
        if self.transform:
            # Transpose to (H, W, C) for Albumentations
            augmented = self.transform(image=image.transpose(1, 2, 0), mask=mask)
            image = augmented['image'].transpose(2, 0, 1) # Back to (C, H, W)
            mask = augmented['mask']
        
        # 5. Final Formatting
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0) # Add channel dim: (1, H, W)
        
        return torch.from_numpy(image), torch.from_numpy(mask)

    def create_mask_from_geojson(self, geojson_path, meta):
        """Helper to create mask on-the-fly if PNG is missing."""
        try:
            gdf = gpd.read_file(geojson_path)
            if gdf.empty:
                return np.zeros((meta['height'], meta['width']), dtype=np.uint8)
            
            # Project -> Buffer (2m) -> Reproject
            gdf = gdf.to_crs(epsg=3857)
            gdf['geometry'] = gdf.geometry.buffer(2)
            gdf = gdf.to_crs(meta['crs'])
            
            # Rasterize
            mask = rasterize(
                [(g, 255) for g in gdf.geometry],
                out_shape=(meta['height'], meta['width']),
                transform=meta['transform'],
                fill=0,
                dtype=np.uint8
            )
            return mask
        except Exception as e:
            print(f"Error generating mask for {geojson_path}: {e}")
            return np.zeros((meta['height'], meta['width']), dtype=np.uint8)