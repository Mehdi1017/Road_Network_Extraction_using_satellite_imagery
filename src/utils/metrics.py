import numpy as np
import networkx as nx
import sknw
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
import hashlib

METERS_PER_PIXEL = 0.3
NUM_CONTROL_POINTS = 50

def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return 1.0 if union == 0 and intersection == 0 else (intersection / union if union > 0 else 0.0)

def width_to_speed_kph(width_m):
    if width_m < 4.0: return 20
    if width_m < 7.0: return 40
    if width_m < 10.0: return 60
    return 90

def assign_speeds(G, mask):
    dist_map = distance_transform_edt(mask)
    width_map_m = (dist_map * 2) * METERS_PER_PIXEL
    for u, v, data in G.edges(data=True):
        if 'pts' in data:
            pts = data['pts']
            ys = np.clip(pts[:, 0], 0, width_map_m.shape[0]-1)
            xs = np.clip(pts[:, 1], 0, width_map_m.shape[1]-1)
            mean_width = np.mean(width_map_m[ys, xs])
        else:
            p1, p2 = G.nodes[u]['o'], G.nodes[v]['o']
            mean_width = (width_map_m[int(p1[0]), int(p1[1])] + width_map_m[int(p2[0]), int(p2[1])]) / 2.0
        
        speed = width_to_speed_kph(mean_width)
        length_km = (data['weight'] * METERS_PER_PIXEL) / 1000.0
        data['travel_time_h'] = length_km / speed
    return G

def mask_to_graph(mask):
    if mask.sum() == 0: return nx.Graph()
    ske = skeletonize(mask).astype(np.uint16)
    G = sknw.build_sknw(ske, multi=False)
    return assign_speeds(G, mask)

def get_closest_node(G, coord):
    if len(G.nodes) == 0: return None
    nodes = list(G.nodes(data=True))
    coords = np.array([d['o'] for n, d in nodes])
    tree = cKDTree(coords)
    dist, idx = tree.query(coord)
    return nodes[idx][0] if dist <= 50 else None

def calculate_apls(G_gt, G_pred, weight='weight'):
    if len(G_gt) < 2 or len(G_pred) < 2: return 0.0
    gt_nodes = list(G_gt.nodes(data=True))
    scores = []
    
    for _ in range(NUM_CONTROL_POINTS):
        idx1, idx2 = np.random.choice(len(gt_nodes), 2, replace=False)
        n1, d1 = gt_nodes[idx1]
        n2, d2 = gt_nodes[idx2]
        
        try: 
            len_gt = nx.shortest_path_length(G_gt, source=n1, target=n2, weight=weight)
        except nx.NetworkXNoPath: 
            continue
            
        if len_gt == 0: continue
        
        np1 = get_closest_node(G_pred, d1['o'])
        np2 = get_closest_node(G_pred, d2['o'])
        
        if not np1 or not np2: 
            scores.append(0.0)
            continue
        
        try: 
            len_pred = nx.shortest_path_length(G_pred, source=np1, target=np2, weight=weight)
        except nx.NetworkXNoPath: 
            scores.append(0.0)
            continue
        
        diff = abs(len_gt - len_pred)
        scores.append(max(0, 1 - (diff / len_gt)))
        
    return np.mean(scores) if scores else 0.0

def wl_subtree_kernel(G1, G2, h=3):
    """
    Weisfeiler-Lehman Subtree Kernel to measure structural graph isomorphism.
    """
    if len(G1.nodes) == 0 or len(G2.nodes) == 0:
        return 0.0
        
    def get_labels(G):
        return {n: str(G.degree(n)) for n in G.nodes()}
        
    labels1 = get_labels(G1)
    labels2 = get_labels(G2)
    all_labels1 = list(labels1.values())
    all_labels2 = list(labels2.values())
    
    for i in range(h):
        new_labels1 = {}; new_labels2 = {}
        for n in G1.nodes():
            nbrs = sorted([labels1[nbr] for nbr in G1.neighbors(n)])
            new_labels1[n] = hashlib.md5((labels1[n] + "".join(nbrs)).encode()).hexdigest()
        for n in G2.nodes():
            nbrs = sorted([labels2[nbr] for nbr in G2.neighbors(n)])
            new_labels2[n] = hashlib.md5((labels2[n] + "".join(nbrs)).encode()).hexdigest()
            
        labels1 = new_labels1; labels2 = new_labels2
        all_labels1.extend(labels1.values())
        all_labels2.extend(labels2.values())
        
    unique = set(all_labels1) | set(all_labels2)
    lmap = {l: i for i, l in enumerate(unique)}
    v1 = np.zeros(len(unique))
    v2 = np.zeros(len(unique))
    
    for l in all_labels1: v1[lmap[l]] += 1
    for l in all_labels2: v2[lmap[l]] += 1
    
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0.0 if norm == 0 else np.dot(v1, v2) / norm