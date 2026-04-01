import numpy as np
import networkx as nx
import math
import sknw
from skimage.morphology import skeletonize, closing, square, remove_small_objects, dilation, disk
from skimage.filters import median
from skimage.draw import line_aa
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw

# --- MORPHOLOGICAL REFINEMENT ---
def morphological_refinement(probs, threshold=0.2, min_size=2000):
    """Heavy morphological refinement (Mega-Connect)."""
    mask = (probs > threshold).astype(np.uint8)
    
    mask = dilation(mask, square(7))
    mask = median(mask, disk(5))
    mask = closing(mask, square(15))
    mask = remove_small_objects(mask.astype(bool), min_size=min_size).astype(np.uint8)
    
    return mask

def base_prep_for_graphs(probs, threshold=0.3):
    """Lighter morphological prep used before Filin/Li vectorization."""
    mask = (probs > threshold).astype(np.uint8) * 255
    mask = dilation(mask, square(3))
    mask = closing(mask, square(5))
    mask = remove_small_objects(mask.astype(bool), min_size=300).astype(np.uint8)
    return mask

# --- FILIN ET AL. HEURISTICS ---
def filin_vectorization(mask, cluster_distance=30):
    y, x = np.where(mask > 0)
    if len(y) < 50: return nx.Graph()
    
    points = np.column_stack((y, x))
    if len(points) > 1000: points = points[::3] 

    estimated_clusters = max(2, int(len(points) / 15)) 
    kmeans = MiniBatchKMeans(n_clusters=estimated_clusters, n_init=3, batch_size=256, random_state=42)
    kmeans.fit(points)
    centroids = kmeans.cluster_centers_
    
    G = nx.Graph()
    for i, c in enumerate(centroids): G.add_node(i, o=c)
    if len(centroids) < 2: return G
    
    tree = cKDTree(centroids)
    for i, c in enumerate(centroids):
        dists, idxs = tree.query(c.reshape(1, -1), k=8, distance_upper_bound=cluster_distance * 2.5)
        
        if isinstance(dists, (float, np.float32, np.float64)): dists, idxs = [dists], [idxs]
        elif len(dists.shape) > 1: dists, idxs = dists[0], idxs[0]
        
        for d, neighbor_idx in zip(dists, idxs):
            if neighbor_idx >= len(centroids) or i == neighbor_idx or d == float('inf'): continue
            pts_array = np.array([c, centroids[neighbor_idx]])
            G.add_edge(i, neighbor_idx, weight=d, pts=pts_array)

    return G

def filin_refinement(G, img_shape=(1300, 1300)):
    if len(G.nodes) < 2: return G
    components = list(nx.connected_components(G))
    nodes_to_remove = []

    for comp in components:
        if len(comp) < 3:
            nodes_to_remove.extend(list(comp))
            continue
            
        subgraph = G.subgraph(comp)
        touches_border, has_intersection = False, False
        
        for n in subgraph.nodes():
            y, x = subgraph.nodes[n]['o']
            if x < 10 or x > img_shape[1]-10 or y < 10 or y > img_shape[0]-10: touches_border = True
            if subgraph.degree(n) > 2: has_intersection = True
                
        if not touches_border and not has_intersection:
            nodes_to_remove.extend(list(comp))

    G.remove_nodes_from(nodes_to_remove)
    return G

# --- LI ET AL. HEURISTICS ---
def get_vector_angle(p1, p2): return math.degrees(math.atan2(p2[0] - p1[0], p2[1] - p1[1]))
def angle_difference(a1, a2): diff = abs(a1 - a2); return min(diff, 360 - diff)

def get_endpoint_direction(G, node_id):
    curr = node_id
    path = [curr]
    for _ in range(8): 
        neighbors = [n for n in G.neighbors(curr) if n not in path]
        if not neighbors: break
        curr = neighbors[0] 
        path.append(curr)
        
    if len(path) < 3: return 0 
    return get_vector_angle(G.nodes[path[-1]]['o'], G.nodes[path[0]]['o'])

def li_postprocessing(mask, connection_radius=80, max_angle_dev=45):
    ske = skeletonize(mask).astype(np.uint16)
    
    for _ in range(2):
        G = sknw.build_sknw(ske, multi=False) 
        endpoints = [n for n in G.nodes() if G.degree(n) == 1]
        if len(endpoints) < 2: break
        
        coords = np.array([G.nodes[n]['o'] for n in endpoints])
        tree = cKDTree(coords)
        used_nodes = set()

        for i, u in enumerate(endpoints):
            if u in used_nodes: continue
            u_pos = coords[i]
            dists, idxs = tree.query(u_pos.reshape(1, -1), k=5, distance_upper_bound=connection_radius)
            if isinstance(dists, (float, np.float32, np.float64)): dists, idxs = [dists], [idxs]
            elif len(dists.shape) > 1: dists, idxs = dists[0], idxs[0]
            
            u_angle = get_endpoint_direction(G, u)
            best_match, min_deviation = None, float('inf')

            for d, j in zip(dists, idxs):
                if j >= len(endpoints): continue
                v = endpoints[j]
                if u == v or d == float('inf') or v in used_nodes: continue

                v_pos = coords[j]
                connect_angle = get_vector_angle(u_pos, v_pos)
                diff_u = angle_difference(u_angle, connect_angle)
                diff_v = angle_difference(get_endpoint_direction(G, v), (connect_angle + 180) % 360)
                
                if diff_u < max_angle_dev and diff_v < max_angle_dev:
                    if (diff_u + diff_v) < min_deviation:
                        min_deviation = (diff_u + diff_v)
                        best_match = (v, u_pos, v_pos)

            if best_match:
                v, start, end = best_match
                rr, cc, val = line_aa(int(start[0]), int(start[1]), int(end[0]), int(end[1]))
                ske[np.clip(rr, 0, ske.shape[0]-1), np.clip(cc, 0, ske.shape[1]-1)] = 1
                used_nodes.update([u, v])

    return sknw.build_sknw(ske, multi=False)

# --- UTILS ---
def save_graph_as_mask(graph, shape, line_width=25):
    """Draws graph edges onto a mask for evaluation."""
    img = Image.new('L', (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(img)
    
    for (s, e) in graph.edges():
        if s not in graph.nodes or e not in graph.nodes: continue
        edge_data = graph[s][e]
        
        if 'pts' in edge_data:
            xy_points = [(p[1], p[0]) for p in edge_data['pts']]
            if len(xy_points) >= 2: draw.line(xy_points, fill=255, width=line_width) 
        else:
            p1, p2 = graph.nodes[s]['o'], graph.nodes[e]['o']
            draw.line([p1[1], p1[0], p2[1], p2[0]], fill=255, width=line_width)
    return np.array(img)