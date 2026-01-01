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
OUTPUT_DIR = "../results/thesis_visuals/graphs_resnet50_improved"

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
        draw_graph_on_image(f, save_path, clean=False)
        
    print(f"Visualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()