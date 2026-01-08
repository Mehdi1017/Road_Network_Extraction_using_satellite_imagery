import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import albumentations as A
import sys

from initial.dataset import RoadDataset
from model import get_model

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Use your best model (ResNet or Transformer)
MODEL_PATH = "../models/unet_mit_b3_best.pth" 
TEST_LIST = "../src/test_list.txt"

# Save to a specific folder so we don't overwrite good results
OUTPUT_DIR = "../results/thesis_visuals/raw_output_mit_b3_earlystopping"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_raw_images():
    print(f"Loading model: {MODEL_PATH}")
    # Update architecture/encoder to match your saved model
    model = get_model('unet', 'mit_b3').to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Transforms (Must match training!)
    test_transform = A.Compose([
        A.PadIfNeeded(min_height=1312, min_width=1312, border_mode=0), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0),
    ])

    test_dataset = RoadDataset(file_list_path=TEST_LIST, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Generating RAW predictions for {len(test_dataset)} images...")

    with torch.no_grad():
        for i, (image, mask) in enumerate(tqdm(test_loader)):
            # Process only first 10 images for visualization
            #if i >= 10: break
            
            image = image.to(DEVICE)
            
            # Forward Pass
            with torch.amp.autocast('cuda'):
                logits = model(image)
                probs = logits.sigmoid()
            
            # --- NO POST-PROCESSING HERE ---
            # Just simple thresholding. This shows the "broken" roads.
            pred_mask = (probs > 0.3).float()
            
            # Crop back to 1300x1300
            pred_np = pred_mask[0, 0, :1300, :1300].cpu().numpy().astype(np.uint8) * 255
            
            # Save
            full_path = test_dataset.image_files[i]
            base_name = os.path.basename(full_path).replace('.tif', '_raw.png')
            Image.fromarray(pred_np).save(os.path.join(OUTPUT_DIR, base_name))

    print(f"Raw images saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_raw_images()