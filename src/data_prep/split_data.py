import argparse
import random
from pathlib import Path

def create_splits(mode: str, split_ratio: float, seed: int, base_dir: Path, raw_dir: Path, splits_dir: Path):
    """
    Core logic to generate dataset text files containing relative paths to the images.
    """
    random.seed(seed)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Dynamically finds all city directories inside data/raw/
    city_paths = [p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("AOI_")]
    city_paths.sort()
    
    if not city_paths:
        print(f"No city folders found in {raw_dir}")
        return
        
    print(f"Found {len(city_paths)} cities: {[c.name for c in city_paths]}")
    
    if mode == "combined":
        all_images = []
        for city_path in city_paths:
            images = list(city_path.glob("PS-RGB/*.tif"))
            images = [img for img in images if not img.name.endswith('.xml')]
            all_images.extend(images)
            
        # Shuffle and split
        random.shuffle(all_images)
        split_idx = int(len(all_images) * split_ratio)
        train_imgs = all_images[:split_idx]
        val_imgs = all_images[split_idx:]
        
        # Writes paths RELATIVE to the project root (e.g., "data/raw/AOI_8_Mumbai/PS-RGB/img.tif")
        with open(splits_dir / "train_list_ALL.txt", "w") as f:
            f.write("\n".join(str(img.relative_to(base_dir)) for img in train_imgs))
            
        with open(splits_dir / "val_list_ALL.txt", "w") as f:
            f.write("\n".join(str(img.relative_to(base_dir)) for img in val_imgs))
            
        print(f"\n✅ Created Combined Lists in {splits_dir.name}/:")
        print(f"   train_list_ALL.txt: {len(train_imgs)} images")
        print(f"   val_list_ALL.txt:   {len(val_imgs)} images")
        
    elif mode == "per_city":
        for city_path in city_paths:
            images = list(city_path.glob("PS-RGB/*.tif"))
            images = [img for img in images if not img.name.endswith('.xml')]
            images.sort() # Ensures reproducible shuffle
            random.shuffle(images)
            
            if not images:
                print(f"No images found for {city_path.name}. Skipping.")
                continue
                
            split_idx = int(len(images) * split_ratio)
            train_imgs = images[:split_idx]
            val_imgs = images[split_idx:]
            
            # Saves specific lists
            with open(splits_dir / f"train_list_{city_path.name}.txt", "w") as f:
                f.write("\n".join(str(img.relative_to(base_dir)) for img in train_imgs))
                
            with open(splits_dir / f"val_list_{city_path.name}.txt", "w") as f:
                f.write("\n".join(str(img.relative_to(base_dir)) for img in val_imgs))
                
            with open(splits_dir / f"test_list_{city_path.name}_full.txt", "w") as f:
                f.write("\n".join(str(img.relative_to(base_dir)) for img in images))
                
            print(f"  ✅ {city_path.name}: {len(train_imgs)} Train | {len(val_imgs)} Val | {len(images)} Test/Full")

def main():
    parser = argparse.ArgumentParser(description="Generate train/val splits for SpaceNet data.")
    
    # User chooses what to generate
    parser.add_argument("--mode", type=str, choices=["combined", "per_city", "both"], default="both",
                        help="Generate combined lists, per-city lists, or both (default).")
    parser.add_argument("--ratio", type=float, default=0.8, help="Training split ratio (default 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible shuffling")
    
    args = parser.parse_args()
    
    # Dynamic Pathing
    # This automatically finds the root of the project
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_dir = base_dir / "data" / "raw"
    splits_dir = base_dir / "data" / "splits"
    
    if args.mode in ["combined", "both"]:
        print("\n--- Generating COMBINED Splits ---")
        create_splits("combined", args.ratio, args.seed, base_dir, raw_dir, splits_dir)
        
    if args.mode in ["per_city", "both"]:
        print("\n--- Generating PER-CITY Splits ---")
        create_splits("per_city", args.ratio, args.seed, base_dir, raw_dir, splits_dir)

if __name__ == "__main__":
    main()