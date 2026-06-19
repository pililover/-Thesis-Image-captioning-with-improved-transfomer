import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphNorm

class CrossAttentionFusionLayer(nn.Module):
    def __init__(self, d_model=1024, nhead=8, use_gate=True):
        super().__init__()
        self.use_gate = use_gate
        # PyTorch MultiheadAttention expects [seq_len, batch, embed_dim] format
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=False)
        
        if self.use_gate:
            # MLP for calculating gate weight dynamically
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            )
            
    def forward(self, query, key_value):
        """
        query: [B, L_q, 1024] - e.g. dense graph features
        key_value: [B, L_kv, 1024] - e.g. image features
        
        MultiheadAttention expects [seq_len, batch, embed_dim], so we transpose before/after
        """
        # Transpose from [B, seq_len, embed] to [seq_len, B, embed]
        query_t = query.transpose(0, 1)          # [L_q, B, embed]
        key_value_t = key_value.transpose(0, 1)  # [L_kv, B, embed]
        
        attn_output, _ = self.cross_attn(
            query=query_t, 
            key=key_value_t, 
            value=key_value_t
        )
        
        # Transpose back from [seq_len, B, embed] to [B, seq_len, embed]
        attn_output = attn_output.transpose(0, 1)
        
        if self.use_gate:
            # Concatenate to decide how much of the original query vs graph context to keep
            gate_input = torch.cat([query, attn_output], dim=-1)
            gate_weight = self.gate(gate_input)
            fused_output = gate_weight * attn_output + (1 - gate_weight) * query
            return fused_output
        
        return attn_output

class GATAdapterLarge(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, num_layers=9, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        self.convs = nn.ModuleList()
        self.graphnorms = nn.ModuleList()
        self.prelus = nn.ModuleList()
        
        # Deep GAT layers with GraphNorm, PReLU, and Residual connections
        for _ in range(num_layers):
            self.convs.append(
                GATConv(in_channels=input_dim, out_channels=hidden_dim, heads=num_heads, concat=False)
            )
            # GraphNorm for stabilizing gradients in deep GNNs (different from BatchNorm - normalizes per graph)
            self.graphnorms.append(GraphNorm(hidden_dim))
            # PReLU to avoid dying ReLU (learnable slope)
            self.prelus.append(nn.PReLU())
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass for the deep GAT Adapter.
        
        Args:
            x: Node features [num_all_nodes, 1024] (can come from a supergraph containing multiple images)
            edge_index: Edge index [2, num_edges]
            batch: Tensor [num_all_nodes] indicating which nodes belong to which image (e.g., [0,0,0,1,1,2,...])
                   Required when processing heterogeneous batches in PyG DataLoader.
                   If None, assume x comes from a single graph.
        
        Returns:
            x: Node features after GAT [num_all_nodes, 1024]
               (NO pooling - keep each node separate so the Decoder can map each node_index)
        """
        for i, (conv, graphnorm, prelu) in enumerate(zip(self.convs, self.graphnorms, self.prelus)):
            # Residual Projection: "highway" for gradients to early layers
            residual = x
            
            # GAT layer
            x = conv(x, edge_index)
            
            # GraphNorm (normalize each graph separately, batch required)
            x = graphnorm(x, batch)
            
            # PReLU activation (learnable slope)
            x = prelu(x)
            
            # Dropout for regularization (disabled during eval via model.eval())
            x = self.dropout(x)
            
            # Residual connection
            x = x + residual
        
        return x
