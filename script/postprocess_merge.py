import os
import torch
import numpy as np
import networkx as nx
import sknw
from torch.utils.data import DataLoader
from skimage.morphology import skeletonize, closing, square, remove_small_objects, dilation
from PIL import Image, ImageDraw
from tqdm import tqdm
import sys
import albumentations as A
from initial.dataset import RoadDataset
from model import get_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "./unet_mit_b3_best.pth" 
TEST_LIST = "../src/test_list.txt"

# Output folders
OUTPUT_DIR = "../results/test_results_mit_b3_merge_postprocess" # Changed folder name to avoid overwriting
MASK_OUTPUT_DIR = "../results/test_results_mit_b3_merge_postprocess"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)


# Merge Parameters
MERGE_THRESHOLD = 15 # Nodes closer than this will be merged
MIN_PATH_LENGTH = 50 # Prune isolated paths shorter than this

def simplify_graph(G, threshold=10):
    """
    Merges nodes that are closer than 'threshold' pixels.
    This acts like a spatial clustering on the graph topology.
    """
    # Create a copy to modify
    H = G.copy()
    
    # 1. Identify short edges
    short_edges = []
    for u, v, data in H.edges(data=True):
        weight = data.get('weight', 0)
        if weight < threshold:
            short_edges.append((u, v, weight))
            
    # Sort by shortest first (greedy merge)
    short_edges.sort(key=lambda x: x[2])
    
    # 2. Merge nodes
    # We use a Union-Find-like logic by contracting edges
    for u, v, w in short_edges:
        if H.has_edge(u, v): # Check if edge still exists (wasn't already merged)
            # Contract v into u
            # nx.contracted_nodes merges v into u. self_loops=False prevents 1-node loops.
            try:
                H = nx.contracted_nodes(H, u, v, self_loops=False)
            except ValueError:
                pass # Skip if nodes invalid
                
    return H

def prune_graph(G, min_len=50):
    """Removes small isolated subgraphs."""
    if len(G.nodes) < 2: return G
    
    components = list(nx.connected_components(G))
    for comp in components:
        subgraph = G.subgraph(comp)
        total_len = sum(d.get('weight', 0) for u, v, d in subgraph.edges(data=True))
        
        if total_len < min_len:
            G.remove_nodes_from(comp)
            
    return G

def save_graph_as_mask(graph, shape, save_path):
    img = Image.new('L', (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(img)
    
    for (s, e) in graph.edges():
        # Get coordinates. 
        # Note: After merging, 'o' (original coord) might be gone or moved.
        # We need robust coordinate retrieval.
        
        # If edge has 'pts', use that.
        edge_data = graph[s][e]
        
        # Handle MultiGraph structure if sknw returns one
        if isinstance(edge_data, dict) and 0 in edge_data: # MultiGraph dictionary
             edge_data = edge_data[0] # Take first key
             
        if 'pts' in edge_data:
            pts = edge_data['pts']
            xy_points = [(p[1], p[0]) for p in pts]
            if len(xy_points) >= 2:
                draw.line(xy_points, fill=255, width=15)
        else:
            # Fallback: Use node coordinates
            # sknw nodes have 'o' attribute = numpy array [y, x]
            if 'o' in graph.nodes[s] and 'o' in graph.nodes[e]:
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

    print(f"Running Node Merge Post-Processing on {len(test_dataset)} images...")

    with torch.no_grad():
        for i, (image, mask) in enumerate(tqdm(test_loader)):
            image = image.to(DEVICE)
            
            with torch.amp.autocast('cuda'):
                logits = model(image)
                probs = logits.sigmoid()
            
            pred_mask = (probs > 0.3).float()
            
            # Crop
            h, w = pred_mask.shape[2], pred_mask.shape[3]
            if h > 1300 and w > 1300:
                pred_np = pred_mask[0, 0, :1300, :1300].cpu().numpy().astype(np.uint8) * 255
            else:
                pred_np = pred_mask[0, 0].cpu().numpy().astype(np.uint8) * 255

            # 1. Base Morphology (Essential baseline)
            pred_np = dilation(pred_np, square(3))
            pred_np = closing(pred_np, square(5))
            pred_np = remove_small_objects(pred_np.astype(bool), min_size=300).astype(np.uint8)
            
            # 2. Extract Graph
            ske = skeletonize(pred_np).astype(np.uint16)
            G = sknw.build_sknw(ske, multi=False)
            
            # 3. Simplify Graph (Merge close nodes)
            G_simple = simplify_graph(G, threshold=MERGE_THRESHOLD)
            
            # 4. Prune
            G_final = prune_graph(G_simple, min_len=MIN_PATH_LENGTH)
            
            # Save
            if hasattr(test_dataset, 'image_files'):
                full_path = test_dataset.image_files[i]
                base_name = os.path.basename(full_path).replace('.tif', '.png')
            else:
                base_name = f"image_{i}_merge.png"
            
            mask_save_path = os.path.join(MASK_OUTPUT_DIR, base_name)
            save_graph_as_mask(G_final, pred_np.shape, mask_save_path)

    print(f"Done! Masks saved to {MASK_OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference()