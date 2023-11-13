import torch
import pickle as pkl
import numpy as np
import torch.nn as nn
from training import get_data, Linear






def test(problem_number, need_ind = False):
    if need_ind:
        model = torch.load(f'model/ind_model/ind_model2_{problem_number}.pt').to('cpu')
    else:
        model = torch.load(f'model/model/model2_{problem_number}.pt').to('cpu')
    train_input, train_label, test_input, test_label = get_data(problem_number, need_ind)
    model.eval()
    with torch.no_grad():
        
        output = model.forward(test_input)
    output = torch.squeeze(output)
    relative_error = torch.abs((output-test_label)/(test_label+0.000001))
    # print(output.size(), test_label.size())
    print(f'problem2_{problem_number}:', torch.mean(relative_error))
    return relative_error


    
def refine_models():
    from training import train
    for i in range(50):
        a = test(i, need_ind=True)
        if torch.mean(a) == 1:
            print(f'bad model {i}')
            train(problem_number=i, need_ind=True)
    return

if __name__=='__main__':
    # refine_models()
    for i in range(10):
        test(problem_number=i)