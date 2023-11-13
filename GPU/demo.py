from training import Linear
import torch
from training import get_data
import pickle as pkl
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# f = open(f'training_data_modified/problem2_0.pkl', 'rb')
# data = pkl.load(f)
# print(data[-1])

# p = Path('training_data_instance/problem2_0.pkl')
# f_save = open(p, 'rb')
# data = pkl.load(f_save)
# print(len(data[0]['eobj']))
a = torch.tensor([[0,1],[2,3]]).reshape(-1)

print(a)