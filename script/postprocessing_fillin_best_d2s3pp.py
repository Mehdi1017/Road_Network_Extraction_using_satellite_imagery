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
from train_d3s2pp import DeepLabV3PlusD3S2PP


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "./d3s2pp_resnet50_best.pth" 
TEST_LIST = "../src/test_list.txt"

# Output folders
OUTPUT_DIR = "../results/test_results_d3s2pp_filin_postprocess" # Changed folder name to avoid overwriting
MASK_OUTPUT_DIR = "../results/test_results_d3s2pp_filin_masks"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)

# Filin et al. Parameters
CLUSTER_DISTANCE = 30 

def filin_vectorization(mask):
    y, x = np.where(mask > 0)
    if len(y) < 50: return nx.Graph()
    
    points = np.column_stack((y, x))
    
    # Reduced subsampling for better curve following
    if len(points) > 1000:
        points = points[::3] 

    estimated_clusters = max(2, int(len(points) / 15)) 
    
    kmeans = MiniBatchKMeans(n_clusters=estimated_clusters, n_init=3, batch_size=256, random_state=42)
    kmeans.fit(points)
    centroids = kmeans.cluster_centers_
    
    G = nx.Graph()
    for i, c in enumerate(centroids):
        G.add_node(i, o=c)
        
    if len(centroids) < 2: return G
    
    tree = cKDTree(centroids)
    
    for i, c in enumerate(centroids):
        dists, idxs = tree.query(c.reshape(1, -1), k=5, distance_upper_bound=CLUSTER_DISTANCE * 2.5)
        
        if isinstance(dists, (float, np.float32, np.float64)):
             dists = [dists]
             idxs = [idxs]
        elif len(dists.shape) > 1:
             dists = dists[0]
             idxs = idxs[0]
        
        for d, neighbor_idx in zip(dists, idxs):
            if neighbor_idx >= len(centroids): continue 
            if i == neighbor_idx: continue
            if d == float('inf'): continue
            
            pts_array = np.array([c, centroids[neighbor_idx]])
            G.add_edge(i, neighbor_idx, weight=d, pts=pts_array)

    return G

def filin_refinement(G):
    if len(G.nodes) < 2: return G
    components = list(nx.connected_components(G))
    for comp in components:
        if len(comp) < 3: 
            G.remove_nodes_from(comp)
    return G

def save_graph_as_mask(graph, shape, save_path):
    """
    Draws THICK graph edges onto a black image to simulate a road mask.
    """
    img = Image.new('L', (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(img)
    
    for (s, e) in graph.edges():
        if s not in graph.nodes or e not in graph.nodes: continue
        
        edge_data = graph[s][e]
        
        if 'pts' in edge_data:
            pts = edge_data['pts']
            # Convert (y,x) to (x,y)
            xy_points = [(p[1], p[0]) for p in pts]
            if len(xy_points) >= 2:
                # WIDTH=25: Thicker lines to ensure good APLS/IoU overlap
                draw.line(xy_points, fill=255, width=25) 
        else:
            p1 = graph.nodes[s]['o']
            p2 = graph.nodes[e]['o']
            draw.line([p1[1], p1[0], p2[1], p2[0]], fill=255, width=25)
        
    img.save(save_path)

def run_inference():
    print(f"Loading model: {MODEL_PATH}")
    model = DeepLabV3PlusD3S2PP(encoder_name="resnet50", classes=1).to(DEVICE)
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
            
            # Crop logic
            h, w = pred_mask.shape[2], pred_mask.shape[3]
            if h > 1300 and w > 1300:
                pred_np = pred_mask[0, 0, :1300, :1300].cpu().numpy().astype(np.uint8) * 255
            else:
                pred_np = pred_mask[0, 0].cpu().numpy().astype(np.uint8) * 255

            # Morphological Prep
            pred_np_morph = dilation(pred_np, square(3))
            pred_np_morph = closing(pred_np_morph, square(5))
            pred_np_morph = remove_small_objects(pred_np_morph.astype(bool), min_size=300).astype(np.uint8)
            
            # Vectorize (Filin)
            graph_filin = filin_vectorization(pred_np_morph)
            graph_filin = filin_refinement(graph_filin)
            
            # Construct Save Path
            if hasattr(test_dataset, 'image_files'):
                full_path = test_dataset.image_files[i]
                base_name = os.path.basename(full_path).replace('.tif', '.png')
            else:
                base_name = f"image_{i}_filin.png"
            
            mask_save_path = os.path.join(MASK_OUTPUT_DIR, base_name)
            
            # Save Mask
            save_graph_as_mask(graph_filin, pred_np.shape, mask_save_path)

    print(f"Done! Evaluation Masks saved to {MASK_OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference()