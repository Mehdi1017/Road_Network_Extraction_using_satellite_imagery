# src/data_prep/preprocess.py
import argparse
from pathlib import Path
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from tqdm import tqdm

def process_city(city_dir: Path):
    """
    Processes a single city directory: finds GeoJSONs, buffers road lines, 
    and rasterizes them into binary PNG masks.
    """
    rgb_dir = city_dir / 'PS-RGB'
    geojson_dir = city_dir / 'geojson_roads'
    mask_dir = city_dir / 'PS-RGB-Masks'
    
    # Validation check
    if not geojson_dir.exists():
        geojson_dir = city_dir / 'geojson_roads_speed'  # Try alternative folder name
    if not rgb_dir.exists() or not geojson_dir.exists():
        print(f"⚠️ Skipping {city_dir.name} - Missing 'PS-RGB' or 'geojson_roads' folders.")
        return

    mask_dir.mkdir(parents=True, exist_ok=True)
    label_files = list(geojson_dir.glob('*.geojson'))
    
    if not label_files:
        print(f"⚠️ No GeoJSON files found in {geojson_dir}")
        return

    print(f"\n🌍 Processing {city_dir.name} | Found {len(label_files)} tiles.")

    for label_path in tqdm(label_files, desc=city_dir.name):
        try:
            # String manipulation for SpaceNet naming conventions
            base_name = label_path.name.replace('geojson_roads', 'PS-RGB').replace('.geojson', '.tif').replace('_speed', '')
            image_path = rgb_dir / base_name
            
            if not image_path.exists():
                continue # Skip if the corresponding satellite image is missing
            
            with rasterio.open(image_path) as src:
                meta = src.meta.copy()
            
            # Create an empty mask default
            mask = np.zeros((meta['height'], meta['width']), dtype=np.uint8)
            
            # Load labels and create mask
            labels_gdf = gpd.read_file(label_path)
            if not labels_gdf.empty:
                # A. Save the original CRS
                original_crs = labels_gdf.crs

                # B. Project to Web Mercator (EPSG:3857) for accurate metric buffering
                labels_proj = labels_gdf.to_crs(epsg=3857)

                # C. Perform the buffer (2 meters)
                labels_proj['geometry'] = labels_proj.geometry.buffer(2)

                # D. Project BACK to the original CRS
                labels_gdf = labels_proj.to_crs(original_crs)
                
                shapes_to_burn = [(geom, 255) for geom in labels_gdf.geometry]
                mask = rasterize(
                    shapes=shapes_to_burn,
                    out_shape=(meta['height'], meta['width']),
                    transform=meta['transform'],
                    fill=0,
                    dtype=np.uint8
                )
            
            # Save the mask as a PNG
            mask_save_path = mask_dir / base_name.replace('.tif', '.png')
            
            meta.update(driver='PNG', count=1, dtype='uint8', nodata=None)
            with rasterio.open(mask_save_path, 'w', **meta) as dst:
                dst.write(mask, 1)
                
        except Exception as e:
            # Catch errors but don't crash the whole multi-city loop
            print(f"Error processing {label_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch preprocess SpaceNet GeoJSONs into raster masks.")
    
    # Default to the data/raw folder at the root of the project
    project_root = Path(__file__).resolve().parent.parent.parent
    default_data_dir = project_root / "data" / "raw"
    
    parser.add_argument("--data_dir", type=str, default=str(default_data_dir),
                        help="Path to the directory containing the city folders (e.g., AOI_2_Vegas, AOI_8_Mumbai)")
    
    args = parser.parse_args()
    data_root = Path(args.data_dir)
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found at {data_root}. Please check your paths.")
        
    print(f"🔍 Scanning for city datasets in: {data_root}")
    
    # Find all subdirectories inside data/raw (e.g., Mumbai, Vegas, Paris, Khartoum)
    city_folders = [f for f in data_root.iterdir() if f.is_dir()]
    
    if not city_folders:
        print("❌ No city folders found.")
        return
        
    for city_dir in city_folders:
        process_city(city_dir)
        
    print("\n✅ All cities preprocessed successfully!")

if __name__ == "__main__":
    main()