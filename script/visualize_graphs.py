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
MASK_DIR = "../../results/test_results_mit_b3_earlystopping/predictions" 
OUTPUT_DIR = "../../results/thesis_visuals/graphs_mit_b3_earlystopping"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def mask_to_graph_vis(mask, clean=False):
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

    # --- FIX FOR PERFECT LOOPS (The "Spur Creator") ---
    # Even with multi=True, sknw needs ONE node to anchor the loop.
    # This ensures that node exists.
    
    kernel = np.ones((3, 3), dtype=int)
    # Convolve to find neighbors. Subtract 1 to ignore the pixel itself.
    # Multiply by ske to ignore background pixels.
    neighbors = (convolve2d(ske, kernel, mode='same') - 1) * ske
    
    labeled_ske, num_features = label(ske, connectivity=2, return_num=True)
    
    for i in range(1, num_features + 1):
        comp_mask = (labeled_ske == i)
        comp_neighbors = neighbors[comp_mask]
        
        # If component is a perfect loop (all 2 neighbors)
        if np.all(comp_neighbors == 2) and len(comp_neighbors) > 10:
            cy, cx = np.where(comp_mask)
            py, px = cy[0], cx[0]
            
            # Find an empty spot to add a spur (Create a Node)
            found_spot = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0: continue
                    ny, nx = py + dy, px + dx
                    
                    if 0 <= ny < ske.shape[0] and 0 <= nx < ske.shape[1]:
                        if ske[ny, nx] == 0:
                            ske[ny, nx] = 1 # Add spur
                            found_spot = True
                            break
                if found_spot: break
    # --------------------------------------------------

    # 3. Build Graph (Enable MultiGraph)
    # multi=True allows self-loops (A->A) and parallel edges
    graph = sknw.build_sknw(ske, multi=True)
    
    return graph, ske

def draw_graph_on_image(mask_path, save_path, clean=False):
    mask = np.array(Image.open(mask_path))
    mask = (mask > 127).astype(np.uint8)
    
    if mask.sum() == 0: return 

    graph, ske = mask_to_graph_vis(mask, clean=clean)
    
    plt.figure(figsize=(12, 12))
    
    # 1. SHOW THE SKELETON (Gray Lines)
    plt.imshow(ske, cmap='gray')
    
    # 2. SHOW THE GRAPH EDGES (Green Lines)
    # Iterate over edges (handling MultiGraph structure)
    for (s, e, k) in graph.edges(keys=True):
        edge_data = graph[s][e][k]
        
        # Check if 'pts' exists (it contains the pixel coordinates)
        if 'pts' in edge_data:
            ps = edge_data['pts']
            plt.plot(ps[:, 1], ps[:, 0], '#00FF00', linewidth=2.5, alpha=0.9) 
        else:
            # Fallback: Draw straight line if no pixel path exists
            # (This handles edges created by refinement or weird sknw cases)
            p1 = graph.nodes[s]['o']
            p2 = graph.nodes[e]['o']
            plt.plot([p1[1], p2[1]], [p1[0], p2[0]], '#00FF00', linewidth=2.5, alpha=0.9, linestyle='--')

    # 3. SHOW THE NODES (Red Dots)
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
        
        # --- FIX: Set clean=False to see raw skeleton/loops ---
        # If running on predictions, you might want True. For Ground Truth debug, False is better.
        draw_graph_on_image(f, save_path, clean=False)
        
    print(f"Visualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()