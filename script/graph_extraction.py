import os
import glob
import numpy as np
import networkx as nx
import sknw
from skimage.morphology import skeletonize, closing, square, remove_small_objects
from scipy.spatial import cKDTree
from PIL import Image
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- CONFIG ---
PRED_DIR = "../results/test_results_d3s2pp_filin_masks"
GT_DIR = "../results/test_results_resnet50_de_parameters/ground_truth"
NUM_CONTROL_POINTS = 50

# CRESI-based parameters (assuming ~30cm/pixel resolution)
# 6 meters ~ 20 pixels
CONNECT_THRESHOLD = 20 
MIN_SUBGRAPH_LENGTH = 50 

def mask_to_graph(mask_path, refine=False):
    """
    Converts a binary mask image into a NetworkX graph with CRESI refinements.
    """
    # 1. Load and Binarize
    mask = np.array(Image.open(mask_path))
    mask = (mask > 127).astype(np.uint8)

    # 2. Pixel Cleaning (CRESI Step: "removing small object artifacts")
    # Remove blobs smaller than ~30 square meters (~300 pixels)
    mask = remove_small_objects(mask.astype(bool), min_size=300).astype(np.uint8)
    
    # 3. Morphological Closing (CRESI Step: "refined mask... closing techniques")
    # 2 meters ~ 7 pixels
    mask = closing(mask, square(7))

    if mask.sum() == 0:
        return nx.Graph()

    # 4. Skeletonize
    ske = skeletonize(mask).astype(np.uint16)

    # 5. Build Initial Graph
    graph = sknw.build_sknw(ske, multi=True)
    
    # 6. Graph Refinement (The "Secret Sauce" of CRESI)
    if refine:
        graph = refine_graph(graph)
    
    return graph

def refine_graph(G):
    """
    Implements the graph connection logic from CRESI paper Section 3.3.
    "Connect terminal vertices if the distance... is less than 6 meters"
    """
    if len(G.nodes) < 2:
        return G

    # 1. Remove tiny isolated subgraphs
    # [cite: 101] "remove disconnected subgraphs with an integrated path length of less than a certain length"
    components = list(nx.connected_components(G))
    for comp in components:
        subgraph = G.subgraph(comp)
        total_len = sum(d['weight'] for u, v, d in subgraph.edges(data=True))
        if total_len < MIN_SUBGRAPH_LENGTH:
            G.remove_nodes_from(comp)

    # 2. Connect Dead Ends (Terminal Vertices)
    # Run multiple passes to close sequential gaps
    for _ in range(2): 
        # Identify degree-1 nodes
        dead_ends = [n for n in G.nodes() if G.degree(n) == 1]
        if not dead_ends: break
            
        # Build a spatial tree of ALL nodes ONCE per pass
        all_nodes = list(G.nodes())
        if len(all_nodes) < 2: break
        
        # Safe coordinate extraction
        # sknw stores coordinates as numpy arrays in the 'o' attribute
        # We need to stack them into a (N, 2) array for cKDTree
        node_coords = {n: d['o'] for n, d in G.nodes(data=True)}
        coords = np.array([node_coords[n] for n in all_nodes])
        
        if len(coords) == 0: break

        tree = cKDTree(coords)
        
        for n_id in dead_ends:
            if n_id not in G: continue # Might have been removed or merged

            # Current node coordinates
            curr_coord = node_coords[n_id]
            
            # Find neighbors within threshold (CONNECT_THRESHOLD = 6 meters)
            # k=5 finds the 5 nearest neighbors
            dists, idxs = tree.query(curr_coord.reshape(1, -1), k=5, distance_upper_bound=CONNECT_THRESHOLD)
            
            # tree.query returns arrays if input is array, flatten them
            dists = dists[0]
            idxs = idxs[0]

            valid_neighbors = []
            for d, i in zip(dists, idxs):
                if i >= len(all_nodes): continue # cKDTree returns out-of-bounds for no match (infinity distance)
                if d == float('inf'): continue

                neighbor_id = all_nodes[i]
                
                # Don't connect to self or existing neighbors
                if neighbor_id == n_id: continue
                if neighbor_id in G[n_id]: continue
                
                valid_neighbors.append((d, neighbor_id))
            
            # If we found a valid connection, add the edge!
            if valid_neighbors:
                # Connect to the closest one
                dist, target_id = min(valid_neighbors, key=lambda x: x[0])
                G.add_edge(n_id, target_id, weight=dist)

    return G

def calculate_apls_metric(G_gt, G_pred):
    """
    Calculates APLS score by comparing path lengths between random control points.
    """
    if len(G_gt.nodes) == 0 or len(G_pred.nodes) == 0:
        return 0.0

    gt_nodes = list(G_gt.nodes(data=True))
    scores = []
    
    # Increase samples for better statistical significance
    samples = NUM_CONTROL_POINTS
    
    for _ in range(samples):
        if len(gt_nodes) < 2: break
        
        # Pick random pair in Ground Truth
        idx1, idx2 = np.random.choice(len(gt_nodes), 2, replace=False)
        n1, data1 = gt_nodes[idx1]
        n2, data2 = gt_nodes[idx2]
        
        # 1. Get GT Distance
        try:
            len_gt = nx.shortest_path_length(G_gt, source=n1, target=n2, weight='weight')
        except nx.NetworkXNoPath:
            continue 

        # 2. Snap to Prediction Graph
        p1_coord = data1['o']
        p2_coord = data2['o']
        
        node_pred_1 = get_closest_node(G_pred, p1_coord)
        node_pred_2 = get_closest_node(G_pred, p2_coord)
        
        if node_pred_1 is None or node_pred_2 is None:
            scores.append(0.0) 
            continue

        # 3. Get Pred Distance
        try:
            len_pred = nx.shortest_path_length(G_pred, source=node_pred_1, target=node_pred_2, weight='weight')
        except nx.NetworkXNoPath:
            scores.append(0.0)
            continue
            
        # APLS Formula
        diff = abs(len_gt - len_pred)
        path_score = max(0, 1 - (diff / len_gt))
        scores.append(path_score)

    if not scores: return 0.0
    return np.mean(scores)

def get_closest_node(G, coord):
    """Finds node in G closest to coord, with a stricter threshold."""
    if len(G.nodes) == 0: return None
    
    # Extract all node coords
    nodes = list(G.nodes(data=True))
    coords = np.array([d['o'] for n, d in nodes])
    
    # Use cKDTree for fast lookup
    tree = cKDTree(coords)
    dist, idx = tree.query(coord)
    
    # Tolerance: 50 pixels (~15 meters)
    if dist > 50: 
        return None
        
    return nodes[idx][0]

def main():
    print("--- 🛣️ STARTING IMPROVED APLS EVALUATION ---")
    print("Applying CRESI Graph Refinement (Connecting Gaps < 6m)...")
    
    pred_files = glob.glob(os.path.join(PRED_DIR, "*.png"))
    total_apls = 0
    count = 0
    
    # Use tqdm to show progress
    for f in tqdm(pred_files):
        filename = os.path.basename(f)
        gt_path = os.path.join(GT_DIR, filename)
        
        if not os.path.exists(gt_path): continue
            
        # Apply refine=True only to predictions to fix model gaps
        # Apply refine=False (or True) to GT depending on how clean your GT is
        G_pred = mask_to_graph(f, refine=True)
        G_gt = mask_to_graph(gt_path, refine=False) 
        
        score = calculate_apls_metric(G_gt, G_pred)
        total_apls += score
        count += 1
        
    avg_apls = total_apls / count if count > 0 else 0
    print(f"\n🏆 FINAL RESULTS (With Graph Refinement)")
    print(f"Evaluated {count} images.")
    print(f"Average APLS Score: {avg_apls:.4f}")

if __name__ == "__main__":
    main()