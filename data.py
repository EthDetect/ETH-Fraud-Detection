import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import networkx as nx
import pandas as pd
import numpy as np
import os
from scipy.sparse import coo_matrix


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
    
    motif_features = np.zeros((num_nodes, 1))  # Example for triangle motifs
    
    triangles = list(nx.triangles(G).values())
    motif_features[:, 0] = triangles 
    
    return torch.tensor(motif_features, dtype=torch.float32)


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    combined_features = torch.cat([out, data.motif_features], dim=1)
    loss = F.cross_entropy(combined_features[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        combined_features = torch.cat([out, data.motif_features], dim=1)
        pred = combined_features.argmax(dim=1)
        correct = pred[data.test_mask] == data.y[data.test_mask]
        accuracy = int(correct.sum()) / len(data.test_mask)
        return accuracy


def load_node_data(folder_path):
    print(f"Loading node data from {folder_path}...")
    node_data = pd.DataFrame(columns=['id', 'name'])
    file_count = 0
    row_count = 0
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {file_path}")
            for chunk in pd.read_csv(file_path, header=None, names=['id', 'name'], chunksize=10000):
                node_data = pd.concat([node_data, chunk], ignore_index=True)
                row_count += chunk.shape[0]
                print(f"Loaded {row_count} rows so far...")
            file_count += 1
    print(f"Finished loading {file_count} node files with {row_count} rows.")
    return node_data


def load_relationship_data(folder_path):
    print(f"Loading relationship data from {folder_path}...")
    edge_data = pd.DataFrame(columns=['source', 'target'])
    file_count = 0
    row_count = 0
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {file_path}")
            for chunk in pd.read_csv(file_path, header=None, names=['source', 'target'], chunksize=10000):
                edge_data = pd.concat([edge_data, chunk], ignore_index=True)
                row_count += chunk.shape[0]
                print(f"Loaded {row_count} rows so far...")
            file_count += 1
    print(f"Finished loading {file_count} relationship files with {row_count} rows.")
    return edge_data


def load_label_data(file_path):
    print(f"Loading label data from {file_path}...")
    label_data = pd.read_csv(file_path, header=None, names=['node_id', 'label'])
    print(f"Loaded {label_data.shape[0]} label rows.")
    return label_data


def load_edge_index(file_path):
    print(f"Loading edge index from {file_path}...")
    edge_df = pd.read_csv(file_path, header=None, names=['source', 'target'])
    edge_index = torch.tensor(edge_df.values.T, dtype=torch.long)
    print(f"Loaded edge index with {edge_df.shape[0]} edges.")
    return edge_index


def convert_to_sparse_matrix(edge_index, num_nodes):
    print(f"Converting edge index to sparse matrix...")
    edge_index_np = edge_index.numpy()
    row = edge_index_np[0]
    col = edge_index_np[1]
    data = np.ones(len(row))
    sparse_matrix = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    print(f"Converted to sparse matrix with shape {sparse_matrix.shape}.")
    return sparse_matrix

def prepare_labels(label_df, num_nodes):
    print(f"Preparing labels tensor for {num_nodes} nodes...")
    labels_tensor = torch.full((num_nodes,), fill_value=-1, dtype=torch.long)
    for _, row in label_df.iterrows():
        if row['node_id'] < num_nodes:
            labels_tensor[row['node_id']] = row['label']
    print(f"Prepared labels tensor with {torch.sum(labels_tensor >= 0)} valid labels.")
    return labels_tensor


node_folder_path = ''
relationship_folder_path = ''
label_file_path = ''
intermediate_edge_file_path = ''


node_data = load_node_data(node_folder_path)
relationship_data = load_relationship_data(relationship_folder_path)
label_data = load_label_data(label_file_path)


relationship_node_ids = pd.unique(relationship_data[['source', 'target']].values.ravel())
label_node_ids = pd.unique(label_data['node_id'])
all_node_ids = pd.unique(np.concatenate([relationship_node_ids, label_node_ids]))


filtered_node_data = node_data[node_data['id'].isin(all_node_ids)]
print(f"Filtered node data to {filtered_node_data.shape[0]} nodes.")


def convert_to_pyg_data(node_df):
    print(f"Converting node data to PyTorch Geometric format...")
    node_df['name'] = node_df['name'].astype('category').cat.codes
    x = torch.tensor(node_df['name'].values, dtype=torch.float32).view(-1, 1)
    node_id_to_index = {node_id: index for index, node_id in enumerate(node_df['id'])}
    print(f"Converted node data with {x.size(0)} nodes and {x.size(1)} features.")
    return x, node_id_to_index

x, node_id_to_index = convert_to_pyg_data(filtered_node_data)


edge_index = load_edge_index(intermediate_edge_file_path)


num_nodes = len(filtered_node_data)
print(f"Number of nodes: {num_nodes}")


sparse_edge_index = convert_to_sparse_matrix(edge_index, num_nodes)


motif_features = extract_motif_features(edge_index, num_nodes)
print(f"Extracted motif features.")


labels_tensor = prepare_labels(label_data, num_nodes)


data = Data(x=x, edge_index=edge_index, y=labels_tensor)
data.motif_features = motif_features
print(f"Created PyTorch Geometric Data object.")


num_nodes = x.size(0)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

indices = torch.randperm(num_nodes)
train_indices = indices[:int(0.8 * num_nodes)]
test_indices = indices[int(0.8 * num_nodes):]

train_mask[train_indices] = True
test_mask[test_indices] = True

data.train_mask = train_mask
data.test_mask = test_mask
print(f"Created train and test masks.")


in_channels = x.size(1)
hidden_channels = 64
out_channels = labels_tensor.max().item() + 1
num_heads = 4

model = GAT(in_channels + 1, hidden_channels, out_channels, num_heads)  # +1 for motif feature
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
print(f"Initialized model and optimizer.")


num_epochs = 100
for epoch in range(num_epochs):
    loss = train(model, data, optimizer)
    accuracy = evaluate(model, data)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
