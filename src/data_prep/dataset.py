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
        
        # --- 2. LOAD IMAGE LIST ---
        if file_list_path:
            # Training/Test mode: Read from text file
            if os.path.exists(file_list_path):
                with open(file_list_path, 'r') as f:
                    self.image_files = [line.strip() for line in f.readlines()]
            else:
                # If file list is missing, try to generate it or fail gracefully
                print(f"Warning: List file not found: {file_list_path}")
                self.image_files = []
                
        elif data_root:
            # This scans ALL cities recursively
            search_path = os.path.join(data_root, "AOI_*", "PS-RGB", "*.tif")
            self.image_files = glob.glob(search_path)
            self.image_files.sort()
            
        else:
            raise ValueError("Must provide either file_list_path or data_root")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. Load Image
        image_path = self.image_files[idx]
        
        try:
            with rasterio.open(image_path) as src:
                image = src.read() # (C, H, W)
                meta = src.meta
        except Exception as e:
            print(f"Error loading image: {image_path}\n{e}")
            return torch.zeros((3, 512, 512)), torch.zeros((1, 512, 512))

        # 2. DYNAMIC MASK LOADING (Supports Any City)
        # Assumes structure: .../AOI_X_City/PS-RGB/image.tif
        # Target:            .../AOI_X_City/PS-RGB-Masks/image.png
        
        # Replace folder name
        mask_path_png = image_path.replace("/PS-RGB/", "/PS-RGB-Masks/")
        # Replace extension
        mask_path_png = mask_path_png.replace('.tif', '.png')
        
        mask = None

        # Strategy A: Look for pre-processed PNG
        if os.path.exists(mask_path_png):
            try:
                mask = np.array(Image.open(mask_path_png))
            except:
                pass 
        else:
            # Strategy B: Empty (If neither PNG nor GeoJSON exists)
            # This happens for test sets with hidden labels
            
            mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)

        # 3. Fixed Normalization (11-bit)
        image = image.astype(np.float32)
        # SpaceNet 11-bit goes up to 2047. Divide to get 0-1 range.
        image = image / 2048.0
            
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
            return np.zeros((meta['height'], meta['width']), dtype=np.uint8)