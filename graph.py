import pandas as pd
import networkx as nx
import os


folder_paths = [
    '',
    '',
    ''
]


G = nx.DiGraph() 


def process_chunk(chunk, folder_name):
    if 'node' in folder_name:
        G.add_nodes_from([(node_id, {'name': name}) for node_id, name in chunk.values])
    elif 'newCross' in folder_name:
        for node_id, label in chunk.values:
            if node_id in G:
                G.nodes[node_id]['label'] = label
    elif 'relationship' in folder_name:
        G.add_edges_from([(source, target) for source, target in chunk.values])


for folder_path in folder_paths:
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            folder_name = os.path.basename(folder_path)
            chunk_size = 5000  
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                process_chunk(chunk, folder_name)


print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())


nx.write_gpickle(G, '')