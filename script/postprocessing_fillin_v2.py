import os
import torch
import numpy as np
import networkx as nx
import sknw
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize, closing, square, remove_small_objects, dilation
from sklearn.cluster import MiniBatchKMeans # Faster than KMeans
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
OUTPUT_DIR = "../results/test_results_mit_b3_filin_postprocess" # Changed folder name to avoid overwriting
MASK_OUTPUT_DIR = "../results/test_results_filin_masks"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)

# Filin et al. Parameters
CLUSTER_DISTANCE = 30
MAX_ANGLE_DEVIATION = 45 # Degrees
PROLONGATION_ANGLE = 10  # Degrees (stricter for gaps)
IMAGE_SIZE = 1300 # Assuming 1300x1300 for border check

def get_angle(p1, p2):
    return math.degrees(math.atan2(p2[0] - p1[0], p2[1] - p1[1]))

def angle_diff(a1, a2):
    diff = abs(a1 - a2)
    return min(diff, 360 - diff)

def filin_vectorization(mask):
    y, x = np.where(mask > 0)
    if len(y) < 50: return nx.Graph()
    
    points = np.column_stack((y, x))
    if len(points) > 1000: points = points[::3] 

    # Dynamic cluster estimation (Filin Section 5.1)
    estimated_clusters = max(2, int(len(points) / 15)) 
    
    kmeans = MiniBatchKMeans(n_clusters=estimated_clusters, n_init=3, batch_size=256, random_state=42)
    kmeans.fit(points)
    centroids = kmeans.cluster_centers_
    
    G = nx.Graph()
    for i, c in enumerate(centroids):
        G.add_node(i, o=c)
        
    if len(centroids) < 2: return G
    
    tree = cKDTree(centroids)
    
    # 1. Basic Connection (Distance)
    for i, c in enumerate(centroids):
        dists, idxs = tree.query(c.reshape(1, -1), k=8, distance_upper_bound=CLUSTER_DISTANCE * 2.5)
        
        if isinstance(dists, (float, np.float32, np.float64)):
             dists, idxs = [dists], [idxs]
        elif len(dists.shape) > 1:
             dists, idxs = dists[0], idxs[0]
        
        for d, neighbor_idx in zip(dists, idxs):
            if neighbor_idx >= len(centroids): continue 
            if i == neighbor_idx: continue
            if d == float('inf'): continue
            
            # Angle Logic (Section 5.1 & 5.3)
            # This is complex to implement perfectly on a raw graph build without history.
            # Simplified: We add the edge if it is close.
            # We rely on refinement to prune bad angles if possible, 
            # or accept that basic distance connection covers most cases.
            
            pts_array = np.array([c, centroids[neighbor_idx]])
            G.add_edge(i, neighbor_idx, weight=d, pts=pts_array)

    return G

def filin_refinement(G, img_shape=(1300, 1300)):
    """
    Implements Section 5.4: Bad roads removing.
    """
    if len(G.nodes) < 2: return G
    
    components = list(nx.connected_components(G))
    nodes_to_remove = []

    for comp in components:
        # Check 1: Too small?
        if len(comp) < 3:
            nodes_to_remove.extend(list(comp))
            continue
            
        # Check 2: Touching Border? (Section 5.4)
        subgraph = G.subgraph(comp)
        touches_border = False
        has_intersection = False
        
        for n in subgraph.nodes():
            y, x = subgraph.nodes[n]['o']
            # Border check (within 10px of edge)
            if x < 10 or x > img_shape[1]-10 or y < 10 or y > img_shape[0]-10:
                touches_border = True
                break
            # Intersection check (Degree > 2)
            if subgraph.degree(n) > 2:
                has_intersection = True
                
        # Filin Rule: Remove if NOT touching border AND NOT having intersection
        if not touches_border and not has_intersection:
            nodes_to_remove.extend(list(comp))

    G.remove_nodes_from(nodes_to_remove)
    return G

def save_graph_as_mask(graph, shape, save_path):
    img = Image.new('L', (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(img)
    
    for (s, e) in graph.edges():
        if s not in graph.nodes or e not in graph.nodes: continue
        
        edge_data = graph[s][e]
        if 'pts' in edge_data:
            pts = edge_data['pts']
            xy_points = [(p[1], p[0]) for p in pts]
            if len(xy_points) >= 2:
                # Fixed width (Section 5.2 approximation)
                draw.line(xy_points, fill=255, width=25) 
        else:
            p1 = graph.nodes[s]['o']
            p2 = graph.nodes[e]['o']
            draw.line([p1[1], p1[0], p2[1], p2[0]], fill=255, width=25)
        
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

    print(f"Generating Filin masks for {len(test_dataset)} images...")

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

            pred_np_morph = dilation(pred_np, square(3))
            pred_np_morph = closing(pred_np_morph, square(5))
            pred_np_morph = remove_small_objects(pred_np_morph.astype(bool), min_size=300).astype(np.uint8)
            
            graph_filin = filin_vectorization(pred_np_morph)
            # Pass image shape for border check
            graph_filin = filin_refinement(graph_filin, img_shape=pred_np.shape)
            
            if hasattr(test_dataset, 'image_files'):
                full_path = test_dataset.image_files[i]
                base_name = os.path.basename(full_path).replace('.tif', '.png')
            else:
                base_name = f"image_{i}_filin.png"
            
            mask_save_path = os.path.join(MASK_OUTPUT_DIR, base_name)
            save_graph_as_mask(graph_filin, pred_np.shape, mask_save_path)

    print(f"Done! Evaluation Masks saved to {MASK_OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference()