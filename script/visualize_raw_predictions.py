import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import albumentations as A
import sys
# Import D3S2PP class if needed
from script.old.train_d3s2pp import DeepLabV3PlusD3S2PP
from initial.dataset import RoadDataset
from model import get_model

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# SELECT MODEL AND CITY HERE
MODEL_TYPE = 'unet_mit_topo' # 'unet_mit' or 'd3s2pp' or 'resnet50'
CITY_NAME = "AOI_5_Khartoum" # "AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum"

# Define Paths based on selection
if MODEL_TYPE == 'unet_mit_topo':
    MODEL_PATH = os.path.join("topo_unet_mit_b3_best.pth")
    ENCODER = 'mit_b3'
elif MODEL_TYPE == 'resnet50':
    MODEL_PATH = os.path.join("unet_resnet50_best_4cities.pth") # Adjust name
    ENCODER = 'resnet50'
elif MODEL_TYPE == 'd3s2pp':
        MODEL_PATH = os.path.join("d3s2pp_resnet50_best_4cities.pth")
        

TEST_LIST = os.path.join(f"../src/test_list_{CITY_NAME}_full.txt")
OUTPUT_DIR = os.path.join(f"../results/raw_predictions_labeled/{MODEL_TYPE}/{CITY_NAME}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_raw_images():
    if MODEL_TYPE == 'd3s2pp':
        model = DeepLabV3PlusD3S2PP(encoder_name="resnet50", classes=1).to(DEVICE)
    else:
        # Update architecture/encoder to match your saved model
        model = get_model('unet', ENCODER).to(DEVICE)
    if MODEL_TYPE == 'unet_mit_topo':
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() 
    
    
    print(f"Loading model: {MODEL_PATH}")
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
            orig_h, orig_w = 1300, 1300
            # Forward Pass
            with torch.amp.autocast('cuda'):
                logits = model(image)
                # 2. FORCE RESIZE to Original Dimensions
                # This fixes the "zoomed in" or "wrong resolution" issue
                # We resize the logits directly to (orig_h, orig_w)
                logits = torch.nn.functional.interpolate(
                    logits, 
                    size=(orig_h, orig_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                
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