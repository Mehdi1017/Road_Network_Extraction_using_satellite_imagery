import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from skimage.morphology import closing, square, remove_small_objects
import albumentations as A  # Don't forget this!

from skimage.morphology import dilation, closing, square, remove_small_objects, disk
from skimage.filters import median

# Import your custom modules
from initial.dataset import RoadDataset
from model import get_model


# Output folders
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "./unet_mit_b3_best.pth" 
TEST_LIST = "../src/test_list.txt"

# Output folders
OUTPUT_DIR = "../results/test_results_mit_b3_earlystopping" # Changed folder name to avoid overwriting
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
GT_DIR = os.path.join(OUTPUT_DIR, "ground_truth")

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)

def run_inference():
    print(f"Loading model: {MODEL_PATH}")
    model = get_model('unet', 'mit_b3').to(DEVICE)
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
            
            #1. Lower Threshold to catch faint pixels
            pred_mask = (probs > 0.2).float() # Lowered to 0.2
            pred_np = pred_mask[0, 0].cpu().numpy().astype(np.uint8)
            gt_np = mask[0, 0, :1300, :1300].numpy().astype(np.uint8) * 255


            # --- HEAVY POST-PROCESSING ---
            
            # Step A: Heavy Dilation (Thicken roads by ~3 pixels on all sides)
            # A 7x7 kernel bridges gaps up to ~7 pixels
            pred_np = dilation(pred_np, square(7)) 
            
            # Step B: Median Filter (Smooths jagged edges)
            # This removes the "hair" that confuses skeletonization
            pred_np = median(pred_np, disk(5))
            
            # Step C: Heavy Closing (Connect larger disconnects)
            pred_np = closing(pred_np, square(15))
            
            # Step D: Remove floating noise
            pred_np = remove_small_objects(pred_np.astype(bool), min_size=2000).astype(np.uint8)
            
            # -----------------------------
            
            # Save...
            pred_np = pred_np * 255
            
            full_path = test_dataset.image_files[i]
            base_name = os.path.basename(full_path).replace('.tif', '.png')

            Image.fromarray(pred_np).save(os.path.join(PRED_DIR, base_name))
            Image.fromarray(gt_np).save(os.path.join(GT_DIR, base_name))

    print(f"Done! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference()