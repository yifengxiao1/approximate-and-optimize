from training import Linear
import torch
from training import get_data

a = torch.load('model/ind_model/ind_model2_11.pt')
for _, param in enumerate(a.parameters()):
    print(param)