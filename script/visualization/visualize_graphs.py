import os
import glob
import numpy as np
import networkx as nx
import sknw
from skimage.morphology import skeletonize, closing, square, remove_small_objects
from skimage.measure import label
from scipy.signal import convolve2d
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIGURATION ---
MASK_DIR = "../results/test_results_resnet50/predictions" 
OUTPUT_DIR = "../results/thesis_visuals/graphs_resnet50_test"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def mask_to_graph_vis(mask, clean=False, refine=True):
    """
    Extracts graph for visualization.
    Uses 'Spur Creator' + 'multi=True' to handle loops robustly.
    """
    if clean:
        mask = remove_small_objects(mask.astype(bool), min_size=300).astype(np.uint8)
        mask = closing(mask, square(7))

    if mask.sum() == 0:
        return nx.MultiGraph(), mask 

    # 2. Skeletonize
    ske = skeletonize(mask).astype(np.uint16)

    # 3. Build Graph (Enable MultiGraph)
    # multi=True allows self-loops (A->A) and parallel edges
    graph = sknw.build_sknw(ske, multi=True)

     # 5. Graph Refinement
    if refine:
        graph = refine_graph(graph)
    
    return graph, ske

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

def draw_graph_on_image(mask_path, save_path, clean=False):
    mask = np.array(Image.open(mask_path))
    mask = (mask > 127).astype(np.uint8)
    
    if mask.sum() == 0: return 

    graph, ske = mask_to_graph_vis(mask, clean=clean)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(ske, cmap='gray')
    
    # Iterate over edges (handling MultiGraph structure)
    for (s, e, k) in graph.edges(keys=True):
        ps = graph[s][e][k]['pts']
        plt.plot(ps[:, 1], ps[:, 0], '#00FF00', linewidth=2.5, alpha=0.9) 

    nodes = graph.nodes()
    ps = np.array([nodes[i]['o'] for i in nodes])
    if len(ps) > 0:
        plt.plot(ps[:, 1], ps[:, 0], 'r.', markersize=8)

    plt.title(f"Graph: {len(graph.nodes())} Nodes, {len(graph.edges())} Edges")
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    files = glob.glob(os.path.join(MASK_DIR, "*.png"))
    print(f"Generating high-contrast graphs with multi=True...")
    
    for f in tqdm(files):
        filename = os.path.basename(f).replace('.png', '_graph_vis.png')
        save_path = os.path.join(OUTPUT_DIR, filename)
        draw_graph_on_image(f, save_path, clean=True)
        
    print(f"Visualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()