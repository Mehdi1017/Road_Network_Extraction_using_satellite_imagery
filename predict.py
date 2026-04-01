import argparse
import torch
import torch.nn.functional as F
import numpy as np
import rasterio
import albumentations as A
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# --- IMPORTS FROM SRC MODULE ---
from src.data_prep.dataset import RoadDataset
from src.models.architectures import get_model
from src.models.modules import DeepLabV3PlusD3S2PP
from src.post_process.heuristics import (
    morphological_refinement, 
    base_prep_for_graphs,
    filin_vectorization, filin_refinement, 
    li_postprocessing, save_graph_as_mask
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Inference & Topological Post-Processing")
    
    parser.add_argument("--test_list", type=str, required=True, help="Path to the test list (e.g., data/splits/test_list_AOI_5_Khartoum_full.txt)")
    parser.add_argument("--model", type=str, required=True, choices=["mit_b3", "resnet50", "d3s2pp"])
    parser.add_argument("--weights", type=str, required=True, help="Path to the trained .pth file")
    parser.add_argument("--post_process", type=str, default="none", choices=["none", "morpho", "filin", "li"], help="Which heuristics to apply")
    
    return parser.parse_args()

def main():
    args = parse_args()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    # Example: results/mit_b3/filin_masks/AOI_5_Khartoum
    city_name = Path(args.test_list).stem.replace("test_list_", "").replace("_full", "")
    output_dir = Path(f"results/{args.model}/{args.post_process}_masks/{city_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model loading
    print(f"Loading {args.model.upper()} weights from {args.weights}...")
    if args.model == "d3s2pp":
        model = DeepLabV3PlusD3S2PP(encoder_name="resnet50", classes=1)
    elif args.model == "mit_b3":
        model = get_model('unet', 'mit_b3')
    elif args.model == "resnet50":
        model = get_model('unet', 'resnet50')

    state_dict = torch.load(args.weights, map_location=DEVICE)
    if 'model' in state_dict: state_dict = state_dict['model']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # Data loader
    test_transform = A.Compose([
        A.PadIfNeeded(min_height=1312, min_width=1312, border_mode=0), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0),
    ])
    
    dataset = RoadDataset(file_list_path=args.test_list, transform=test_transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"Processing {len(dataset)} images with '{args.post_process}' post-processing...")

    # Inference loop
    with torch.no_grad():
        for i, (image, _) in enumerate(tqdm(loader)):
            image = image.to(DEVICE)
            
            # Gets original dimensions from rasterio (Fallback to 1300 if missing)
            orig_h, orig_w = 1300, 1300
            file_path = dataset.image_files[i] if hasattr(dataset, 'image_files') else f"image_{i}.png"
            base_name = Path(file_path).name.replace('.tif', '.png')
            
            try:
                with rasterio.open(file_path) as src:
                    orig_h, orig_w = src.height, src.width
            except: pass

                
            # Forward pass and resize to fix "zoomed in" bug on d3s2pp
            with torch.amp.autocast('cuda'):
                logits = model(image)
                logits = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                probs = logits.sigmoid()
                probs_np = probs[0, 0].cpu().numpy()

            # --- APPLY SELECTED POST-PROCESSING ---
            if args.post_process == "morpho":
                final_mask = morphological_refinement(probs_np) * 255
                
            elif args.post_process == "filin":
                prep_mask = base_prep_for_graphs(probs_np)
                graph = filin_vectorization(prep_mask)
                graph = filin_refinement(graph, img_shape=(orig_h, orig_w))
                final_mask = save_graph_as_mask(graph, (orig_h, orig_w), line_width=25)
                
            elif args.post_process == "li":
                prep_mask = base_prep_for_graphs(probs_np)
                graph = li_postprocessing(prep_mask)
                final_mask = save_graph_as_mask(graph, (orig_h, orig_w), line_width=15)
                
            else: # "none"
                final_mask = (probs_np > 0.5).astype(np.uint8) * 255

            # Save
            save_path = output_dir / base_name
            Image.fromarray(final_mask.astype(np.uint8)).save(save_path)

    print(f"Finished! Masks saved to: {output_dir}")

if __name__ == "__main__":
    main()