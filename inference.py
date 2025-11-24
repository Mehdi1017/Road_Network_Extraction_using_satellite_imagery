import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

# Import your custom modules
from dataset import RoadDataset
from model import get_model
from skimage.morphology import closing, square, remove_small_objects # <--- Add this

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Update this to your specific saved model
MODEL_PATH = "unet_resnet50_final.pth" 
TEST_LIST = "test_list.txt"

# Output folders
OUTPUT_DIR = "test_results"
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
GT_DIR = os.path.join(OUTPUT_DIR, "ground_truth")

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)

def run_inference():
    # 1. Load Model
    print(f"Loading model: {MODEL_PATH}")
    model = get_model('unet', 'resnet50').to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() # CRITICAL: Disables dropout/batchnorm for consistent results

    # 2. Load Test Data
    # transform=None because we don't want to flip/rotate test images
    test_dataset = RoadDataset(file_list_path=TEST_LIST, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Running inference on {len(test_dataset)} images...")

    # 3. Processing Loop
    with torch.no_grad():
        for i, (image, mask) in enumerate(tqdm(test_loader)):
            image = image.to(DEVICE)
            
            # Forward Pass
            # Use mixed precision for speed
            with torch.amp.autocast('cuda'):
                logits = model(image)
                probs = logits.sigmoid() # Convert logits to 0.0-1.0
            
            # Binarize (Threshold at 0.5)
            pred_mask = (probs > 0.3).float()
            pred_np = pred_mask[0, 0].cpu().numpy().astype(bool) # Must be boolean for morphology
            
            # 2. Morphological Closing (Connect gaps)
            pred_np = closing(pred_np, square(5))
            
            # 3. REMOVE SMALL OBJECTS (The Island Fix) 🏝️
            # Removes any isolated blob smaller than 500 pixels
            pred_np = remove_small_objects(pred_np, min_size=500)
            
            # Prepare for saving
            pred_np = pred_np.astype(np.uint8) * 255
            gt_np = mask[0, 0].numpy().astype(np.uint8) * 255

            # --- SAVE RESULTS ---
            # We need the original filename to keep things organized
            # The dataset stores the full path, let's extract the basename
            full_path = test_dataset.image_files[i]
            base_name = os.path.basename(full_path).replace('.tif', '.png')

            # Save Prediction
            Image.fromarray(pred_np).save(os.path.join(PRED_DIR, base_name))
            
            # Save Ground Truth (Reference for APLS)
            Image.fromarray(gt_np).save(os.path.join(GT_DIR, base_name))

    print(f"Done! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference()