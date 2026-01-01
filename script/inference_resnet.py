import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from skimage.morphology import closing, square, remove_small_objects
import albumentations as A  # Don't forget this!

# Import your custom modules
from dataset import RoadDataset
from model import get_model

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "../models/unet_resnet50_final.pth" 
TEST_LIST = "../src/test_list.txt"

# Output folders
OUTPUT_DIR = "../results/test_results_resnet" # Changed folder name to avoid overwriting
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
GT_DIR = os.path.join(OUTPUT_DIR, "ground_truth")

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)

def run_inference():
    print(f"Loading model: {MODEL_PATH}")
    model = get_model('unet', 'resnet50').to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # --- FIX 1: ADD TRANSFORMS ---
    # The model was trained with Normalization, so we must use it here too.
    # We also need Padding if the image isn't divisible by 32 (standard practice).
    test_transform = A.Compose([
        A.PadIfNeeded(min_height=1312, min_width=1312, border_mode=0), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0),
    ])

    test_dataset = RoadDataset(file_list_path=TEST_LIST, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Running inference on {len(test_dataset)} images...")

    with torch.no_grad():
        for i, (image, mask) in enumerate(tqdm(test_loader)):
            image = image.to(DEVICE)
            
            # Forward Pass
            with torch.amp.autocast('cuda'):
                logits = model(image)
                probs = logits.sigmoid()
            
            # --- POST-PROCESSING ---
            
            # 1. Binarize (Low threshold for ResNet is usually good)
            pred_mask = (probs > 0.3).float()
            
            # Crop back to original size (remove padding)
            # Assuming images are 1300x1300
            pred_np = pred_mask[0, 0, :1300, :1300].cpu().numpy().astype(bool)
            gt_np = mask[0, 0, :1300, :1300].numpy().astype(np.uint8) * 255
            
            # 2. Morphological Closing (Bridge small gaps)
            pred_np = closing(pred_np, square(5))
            
            # 3. REMOVE SMALL OBJECTS (The Fix) 🧹
            # Reduced from 2000 to 100.
            # This keeps small road segments but deletes single-pixel noise.
            pred_np = remove_small_objects(pred_np, min_size=100)
            
            # ---------------------
            
            # Save
            pred_np = pred_np.astype(np.uint8) * 255
            
            full_path = test_dataset.image_files[i]
            base_name = os.path.basename(full_path).replace('.tif', '.png')

            Image.fromarray(pred_np).save(os.path.join(PRED_DIR, base_name))
            Image.fromarray(gt_np).save(os.path.join(GT_DIR, base_name))

    print(f"Done! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference()