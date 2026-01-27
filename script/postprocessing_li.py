import os
import torch
import numpy as np
import networkx as nx
import sknw
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize, closing, square, remove_small_objects, dilation
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import sys
import albumentations as A
from initial.dataset import RoadDataset
from model import get_model



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "./unet_mit_b3_best.pth" 
TEST_LIST = "../src/test_list.txt"

# Output folders
OUTPUT_DIR = "../results/test_results_mit_b3_li_postprocess" # Changed folder name to avoid overwriting
MASK_OUTPUT_DIR = "../results/test_results_mit_b3_li_masks"
VIS_OUTPUT_DIR = "../results/test_results_li_visuals"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)

# Optimized Parameters
CONNECTION_RADIUS = 80 
MAX_ANGLE_DEV = 45 

def get_vector_angle(p1, p2):
    return math.degrees(math.atan2(p2[0] - p1[0], p2[1] - p1[1]))

def angle_difference(a1, a2):
    diff = abs(a1 - a2)
    return min(diff, 360 - diff)

def get_endpoint_direction(G, node_id):
    curr = node_id
    path = [curr]
    
    # Look back further to get a stable direction (avg over 8 pixels)
    for _ in range(8): 
        neighbors = list(G.neighbors(curr))
        valid = [n for n in neighbors if n not in path]
        if not valid: break
        curr = valid[0] 
        path.append(curr)
        
    if len(path) < 3: return 0 
    
    # Vector from internal node -> endpoint
    p_end = G.nodes[path[0]]['o']
    p_internal = G.nodes[path[-1]]['o'] 
    
    return get_vector_angle(p_internal, p_end)

def li_postprocessing(mask):
    # 1. Base Morphological Cleaning
    mask = dilation(mask, square(3)) 
    mask = closing(mask, square(5))
    mask = remove_small_objects(mask.astype(bool), min_size=300).astype(np.uint8)

    # 2. Skeletonize
    ske = skeletonize(mask).astype(np.uint16)
    
    # 3. Iterative Connection
    for _ in range(2):
        G = sknw.build_sknw(ske, multi=False) 
        endpoints = [n for n in G.nodes() if G.degree(n) == 1]
        
        if len(endpoints) < 2: break
        
        node_coords = {n: G.nodes[n]['o'] for n in endpoints}
        coords = np.array([node_coords[n] for n in endpoints])
        tree = cKDTree(coords)
        
        added_edges = []
        used_nodes = set()

        for i, u in enumerate(endpoints):
            if u in used_nodes: continue
            u_pos = coords[i]
            
            dists, idxs = tree.query(u_pos.reshape(1, -1), k=5, distance_upper_bound=CONNECTION_RADIUS)
            
            if isinstance(dists, (float, np.float32, np.float64)): dists, idxs = [dists], [idxs]
            elif len(dists.shape) > 1: dists, idxs = dists[0], idxs[0]
            
            u_angle = get_endpoint_direction(G, u)
            
            best_match = None
            min_deviation = float('inf')

            for d, j in zip(dists, idxs):
                if j >= len(endpoints): continue
                v = endpoints[j]
                if u == v or d == float('inf'): continue
                if v in used_nodes: continue

                v_pos = coords[j]
                connect_vector_angle = get_vector_angle(u_pos, v_pos)
                
                # Check 1: Alignment with U
                diff_u = angle_difference(u_angle, connect_vector_angle)
                
                # Check 2: Alignment with V (Incoming)
                v_angle = get_endpoint_direction(G, v)
                diff_v = angle_difference(v_angle, (connect_vector_angle + 180) % 360)
                
                # Combined deviation check
                if diff_u < MAX_ANGLE_DEV and diff_v < MAX_ANGLE_DEV:
                    total_dev = diff_u + diff_v
                    if total_dev < min_deviation:
                        min_deviation = total_dev
                        best_match = (v, d, u_pos, v_pos)

            if best_match:
                v, d, start, end = best_match
                # Draw strict line
                rr, cc, val = line_aa(int(start[0]), int(start[1]), int(end[0]), int(end[1]))
                rr = np.clip(rr, 0, ske.shape[0]-1)
                cc = np.clip(cc, 0, ske.shape[1]-1)
                ske[rr, cc] = 1
                
                used_nodes.add(u)
                used_nodes.add(v)

    # 4. Final Graph Build
    G_final = sknw.build_sknw(ske, multi=False)
    return G_final

from skimage.draw import line_aa

def save_graph_as_mask(graph, shape, save_path):
    img = Image.new('L', (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(img)
    
    for (s, e) in graph.edges():
        edge_data = graph[s][e]
        if 'pts' in edge_data:
            pts = edge_data['pts']
            xy_points = [(p[1], p[0]) for p in pts]
            if len(xy_points) >= 2:
                # Keep width 15 to ensure IoU overlap is good
                draw.line(xy_points, fill=255, width=15) 
        else:
            p1 = graph.nodes[s]['o']
            p2 = graph.nodes[e]['o']
            draw.line([p1[1], p1[0], p2[1], p2[0]], fill=255, width=15)
        
    img.save(save_path)

def run_inference():
    print(f"Loading model: {MODEL_PATH}")
    model = get_model('unet', 'mit_b3').to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return
    model.eval()

    test_transform = A.Compose([
        A.PadIfNeeded(min_height=1312, min_width=1312, border_mode=0), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0),
    ])

    try:
        test_dataset = RoadDataset(file_list_path=TEST_LIST, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    print(f"Running Stricter Li et al. Inference on {len(test_dataset)} images...")

    with torch.no_grad():
        for i, (image, mask) in enumerate(tqdm(test_loader)):
            image = image.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                logits = model(image)
                probs = logits.sigmoid()
            
            pred_mask = (probs > 0.3).float()
            
            h, w = pred_mask.shape[2], pred_mask.shape[3]
            if h > 1300 and w > 1300:
                pred_np = pred_mask[0, 0, :1300, :1300].cpu().numpy().astype(np.uint8) * 255
            else:
                pred_np = pred_mask[0, 0].cpu().numpy().astype(np.uint8) * 255

            # Run Stricter Li Post-Processing
            graph_li = li_postprocessing(pred_np)
            
            if hasattr(test_dataset, 'image_files'):
                full_path = test_dataset.image_files[i]
                base_name = os.path.basename(full_path).replace('.tif', '.png')
            else:
                base_name = f"image_{i}_li.png"
            
            mask_save_path = os.path.join(MASK_OUTPUT_DIR, base_name)
            save_graph_as_mask(graph_li, pred_np.shape, mask_save_path)

    print(f"Done! Masks saved to {MASK_OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference()