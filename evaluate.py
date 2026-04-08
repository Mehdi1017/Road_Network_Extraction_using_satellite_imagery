import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Import math from your library
from src.utils.metrics import (
    calculate_iou, mask_to_graph, 
    calculate_apls, wl_subtree_kernel
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Road Network Predictions Across All Cities")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., mit_b3, d3s2pp)")
    parser.add_argument("--post_process", type=str, required=True, choices=["none", "morpho", "filin", "li"], help="Post-processing used")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Dynamic Paths
    base_dir = Path(__file__).resolve().parent
    splits_dir = base_dir / "data" / "splits"
    raw_dir = base_dir / "data" / "raw"
    
    # e.g., results/mit_b3/filin_masks/
    pred_base_dir = base_dir / "results" / args.model / f"{args.post_process}_masks"
    
    if not pred_base_dir.exists():
        print(f"Prediction folder not found: {pred_base_dir}")
        print("Make sure you run predict.py first!")
        return

    # Find all test lists
    test_lists = list(splits_dir.glob("test_list_*_full.txt"))
    if not test_lists:
        print(f"No test lists found in {splits_dir}")
        return

    results = []
    print(f"Starting Multi-Metric Evaluation for {args.model.upper()} + {args.post_process.upper()} across {len(test_lists)} cities...")

    for test_list in test_lists:
        city_name = test_list.name.replace("test_list_", "").replace("_full.txt", "")
        
        pred_city_dir = pred_base_dir / city_name
        if not pred_city_dir.exists():
            print(f"Skipping {city_name}: No predictions found in {pred_city_dir}")
            continue
            
        print(f"\nEvaluating {city_name}...")
        
        with open(test_list, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
            
        metrics = {'iou': 0, 'apls_len': 0, 'apls_time': 0, 'wl': 0}
        count = 0
        
        for img_rel_path in tqdm(image_paths):
            # Paths
            basename = Path(img_rel_path).name.replace('.tif', '.png')
            pred_path = pred_city_dir / basename
            #print(pred_path)
            
            # Locate Ground Truth using the relative path from the text file
            # E.g., data/raw/AOI_8_Mumbai/PS-RGB/img.tif -> data/raw/AOI_8_Mumbai/PS-RGB-Masks/img.png
            gt_path = base_dir / img_rel_path.replace("/PS-RGB/", "/PS-RGB-Masks/").replace(".tif", ".png")
            #print(gt_path)
            
            if not pred_path.exists() or not gt_path.exists(): 
                continue
            
            # Load Masks
            mask_pred = np.array(Image.open(pred_path)) > 0
            mask_gt = np.array(Image.open(gt_path)) > 0
            
            # Build Graphs (Calculates Speed internally via your metrics.py)
            G_pred = mask_to_graph(mask_pred)
            G_gt = mask_to_graph(mask_gt)
            
            # Calculate 4 Metrics
            metrics['iou'] += calculate_iou(mask_pred, mask_gt)
            metrics['apls_len'] += calculate_apls(G_gt, G_pred, weight='weight')
            metrics['apls_time'] += calculate_apls(G_gt, G_pred, weight='travel_time_h')
            metrics['wl'] += wl_subtree_kernel(G_gt, G_pred, h=3)
            
            count += 1
        print(count)    
        if count > 0:
            avgs = {k: (v / count) for k, v in metrics.items()}
            print(f"{city_name}: IoU={avgs['iou']:.3f} | APLS(Len)={avgs['apls_len']:.3f} | APLS(Time)={avgs['apls_time']:.3f} | WL={avgs['wl']:.3f}")
            
            results.append({
                "City": city_name,
                "Model": args.model,
                "PostProcess": args.post_process,
                "IoU": avgs['iou'],
                "APLS_Length": avgs['apls_len'],
                "APLS_Time": avgs['apls_time'],
                "WL_Kernel": avgs['wl'],
                "Images": count
            })

    # Save Final Report
    if results:
        df = pd.DataFrame(results)
        print("\nFINAL REPORT")
        print(df.to_string(index=False))
        
        save_dir = base_dir / "results" / "reports"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"evaluation_{args.model}_{args.post_process}.csv"
        df.to_csv(save_path, index=False)
        print(f"\nReport saved to {save_path.relative_to(base_dir)}")

if __name__ == "__main__":
    main()