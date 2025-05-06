import torch
import random
from torch_geometric.datasets import DBP15K


root_dir = ''

dataset = DBP15K(root=root_dir, pair='en_zh')

data = dataset[0]

print("Data attributes:")
for attr in dir(data):
    if not attr.startswith('_'):
        print(f"{attr}: {getattr(data, attr)}")

def sparse_type():
    print(random.randint(70, 90))