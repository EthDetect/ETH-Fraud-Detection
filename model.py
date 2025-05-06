import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=0.6)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
    
def extract_motif_features(edge_index, num_nodes):
    edge_index_np = edge_index.numpy()
    G = nx.Graph()
    G.add_edges_from(edge_index_np.T)
    
    motif_features = np.zeros((num_nodes, 1))  
    triangles = list(nx.triangles(G).values())
    motif_features[:, 0] = triangles 
    
    return torch.tensor(motif_features, dtype=torch.float32)