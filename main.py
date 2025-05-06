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
    

    motif_features = np.zeros((num_nodes, 1)) 
    

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
    node_data = pd.DataFrame(columns=['id', 'name'])
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            for chunk in pd.read_csv(file_path, header=None, names=['id', 'name'], chunksize=10000):
                node_data = pd.concat([node_data, chunk], ignore_index=True)
                
    return node_data


def load_relationship_data(folder_path):
    edge_data = pd.DataFrame(columns=['source', 'target'])
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            for chunk in pd.read_csv(file_path, header=None, names=['source', 'target'], chunksize=10000):
                edge_data = pd.concat([edge_data, chunk], ignore_index=True)
                
    return edge_data


def load_label_data(file_path):
    label_data = pd.read_csv(file_path, header=None, names=['node_id', 'label'])
    return label_data

node_folder_path = ''
relationship_folder_path = ''
label_file_path = ''
intermediate_edge_file_path = ''

def convert_to_pyg_data(node_df):
    node_df['name'] = node_df['name'].astype('category').cat.codes
    x = torch.tensor(node_df['name'].values, dtype=torch.float32).view(-1, 1)
    node_id_to_index = {node_id: index for index, node_id in enumerate(node_df['id'])}
    return x, node_id_to_index
def load_edge_index(file_path):
    edge_df = pd.read_csv(file_path, header=None, names=['source', 'target'])
    return torch.tensor(edge_df.values.T, dtype=torch.long)
def convert_to_sparse_matrix(edge_index, num_nodes):
    edge_index_np = edge_index.numpy()
    row = edge_index_np[0]
    col = edge_index_np[1]
    data = np.ones(len(row))  
    

    sparse_matrix = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    return sparse_matrix
def prepare_labels(label_df, num_nodes):
    labels_tensor = torch.full((num_nodes,), fill_value=-1, dtype=torch.long)
    for _, row in label_df.iterrows():
        if row['node_id'] < num_nodes:
            labels_tensor[row['node_id']] = row['label']
    return labels_tensor

print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')


node_data = load_node_data(node_folder_path)
relationship_data = load_relationship_data(relationship_folder_path)
label_data = load_label_data(label_file_path)

relationship_node_ids = pd.unique(relationship_data[['source', 'target']].values.ravel())
label_node_ids = pd.unique(label_data['node_id'])
all_node_ids = pd.unique(np.concatenate([relationship_node_ids, label_node_ids]))


filtered_node_data = node_data[node_data['id'].isin(all_node_ids)]


x, node_id_to_index = convert_to_pyg_data(filtered_node_data)


edge_index = load_edge_index(intermediate_edge_file_path)


num_nodes = len(filtered_node_data)


sparse_edge_index = convert_to_sparse_matrix(edge_index, num_nodes)


motif_features = extract_motif_features(edge_index, num_nodes)


labels_tensor = prepare_labels(label_data, num_nodes)


data = Data(x=x, edge_index=edge_index, y=labels_tensor)
data.motif_features = motif_features


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


in_channels = x.size(1)
hidden_channels = 64
out_channels = labels_tensor.max().item() + 1
num_heads = 4

model = GAT(in_channels + 1, hidden_channels, out_channels, num_heads)  # +1 for motif feature
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


num_epochs = 100
for epoch in range(num_epochs):
    loss = train(model, data, optimizer)
    accuracy = evaluate(model, data)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')