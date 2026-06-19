import json
import torch
import os
from torch_geometric.data import Data
from tqdm import tqdm

def process_vivg_graph(item):
    """
    Chuyển đổi 1 item JSON từ ViVG-10K thành đối tượng Graph (nút, cạnh, Super Node)
    """
    global_scene_graph = item.get('global_scene_graph', {})
    nodes = global_scene_graph.get('nodes', [])
    relationships = global_scene_graph.get('relationships', [])

    node_texts = []
    node_bboxes = []   # [cx, cy, w, h] normalized to [0, 1] by dividing by 1000
    node_id_to_idx = {}

    # Reference size for normalization — Visual Genome images are ≤ 1024px;
    # dividing by 1000 keeps most values in [0, 1] without needing actual image dims.
    _REF = 1000.0
    
    # Concatenate canonical + attributes (e.g., "large white polar bear") to create node text input for GNN
    for idx, node in enumerate(nodes):
        node_id_to_idx[node['node_id']] = idx
        canonical = node.get('canonical', '')
        attributes = " ".join(node.get('attributes', []))
        
        # Example: "large white polar bear"
        phrase = f"{attributes} {canonical}".strip()
        node_texts.append(phrase)

        # Spatial encoding: [x, y, w, h] → normalized [cx, cy, nw, nh]
        bbox = node.get('bbox', [0, 0, 0, 0])
        x, y, w, h = (bbox + [0, 0, 0, 0])[:4]
        cx = min(max((x + w / 2) / _REF, 0.0), 1.0)
        cy = min(max((y + h / 2) / _REF, 0.0), 1.0)
        nw = min(max(w / _REF, 0.0), 1.0)
        nh = min(max(h / _REF, 0.0), 1.0)
        node_bboxes.append([cx, cy, nw, nh])
        
    num_nodes = len(node_texts)
    
    # Adjacency Matrix
    # only keep original nodes so the Decoder can map each node separately
    edge_list = []
    for rel in relationships:
        sub_id = rel.get('sub')
        obj_id = rel.get('obj')
        if sub_id in node_id_to_idx and obj_id in node_id_to_idx:
            u = node_id_to_idx[sub_id]
            v = node_id_to_idx[obj_id]
            edge_list.append([u, v])
        
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # node_bboxes: [num_nodes, 4] float tensor of normalized [cx, cy, w, h]
    node_bboxes_tensor = torch.tensor(node_bboxes, dtype=torch.float32) if node_bboxes \
        else torch.zeros(num_nodes, 4)

    regions = item.get('regions_mapping', {})
    if isinstance(regions, dict) and len(regions) > 0:
        first_region = list(regions.values())[0]
    elif isinstance(regions, list) and len(regions) > 0:
        first_region = regions[0]
    else:
        first_region = {}
        
    caption_vi = first_region.get('text_vi', '')
    caption_en = first_region.get('text_en', '')
    
    return {
        "image_id": item["image_id"],
        "node_texts": node_texts,
        "node_bboxes": node_bboxes_tensor,
        "edge_index": edge_index,
        "num_nodes": num_nodes,
        "node_id_to_idx": node_id_to_idx,  
        "caption_vi": caption_vi,
        "caption_en": caption_en
    }
