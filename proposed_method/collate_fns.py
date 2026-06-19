import torch

class ViVGCollate:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        labels = torch.stack([item["labels"] for item in batch])
        
        # Concatenate all node_input_ids from all graphs
        node_input_ids_list = []
        node_bboxes_list = []
        edge_index_list = []
        batch_indices = []
        region_node_mask_list = []
        
        node_offset = 0
        for graph_idx, item in enumerate(batch):
            node_ids = item["node_input_ids"]  # [n_i, seq_len]
            edge_idx = item["edge_index"]  # [2, e_i]
            
            n_i = node_ids.size(0)
            
            # Add node input IDs
            node_input_ids_list.append(node_ids)
            node_bboxes_list.append(item["node_bboxes"])  # [n_i, 4]
            
            # Offset edge indices (because we concatenate the graphs)
            if edge_idx.size(1) > 0:
                edge_idx_offset = edge_idx + node_offset
                edge_index_list.append(edge_idx_offset)
            
            # Create batch indices for this graph
            batch_indices.extend([graph_idx] * n_i)
            
            # Region node mask for this sample (flat, same ordering as nodes)
            region_node_mask_list.append(item["region_node_mask"])  # [n_i] bool
            
            node_offset += n_i
        
        # Concatenate
        batched_node_inputs = torch.cat(node_input_ids_list, dim=0)  # [total_nodes, seq_len]
        batched_node_bboxes = torch.cat(node_bboxes_list, dim=0)     # [total_nodes, 4]
        
        if len(edge_index_list) > 0:
            batched_edge_index = torch.cat(edge_index_list, dim=1)  # [2, total_edges]
        else:
            batched_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        batched_batch_tensor = torch.tensor(batch_indices, dtype=torch.long)  # [total_nodes]
        
        # Get pixel_values from preprocessed images in the dataset
        pixel_values = torch.stack([item["pixel_values"] for item in batch])

        region_bboxes = torch.stack([item["region_bbox"] for item in batch])  # [B, 4]
        batched_region_node_mask = torch.cat(region_node_mask_list, dim=0)     # [total_nodes] bool

        return {
            "node_input_ids": batched_node_inputs,       # [total_nodes, seq_len]
            "node_bboxes": batched_node_bboxes,          # [total_nodes, 4]
            "edge_index": batched_edge_index,            # [2, total_edges]
            "batch": batched_batch_tensor,               # [total_nodes]
            "region_bbox": region_bboxes,                # [B, 4]
            "region_node_mask": batched_region_node_mask,# [total_nodes] bool
            "labels": labels,
            "pixel_values": pixel_values
        }
