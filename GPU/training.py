import torch
import pickle as pkl
import numpy as np
import torch.nn as nn
from pathlib import Path


class Linear(nn.Module):
    def __init__(self, x_len):
        super(Linear, self).__init__()
        self.x_len = x_len
        # self.scenario_len = scenario_len
        self.fcx1 = nn.Sequential(nn.Linear(self.x_len, self.x_len), nn.ReLU())
        self.fcx2 = nn.Sequential(nn.Linear(self.x_len, 1), nn.ReLU())
        # self.scenario1 = nn.Sequential(nn.Linear(self.scenario_len, 256), nn.ReLU())
        # self.scenario2 = nn.Sequential(nn.Linear(256, 16), nn.ReLU())
        # self.scenario3 = nn.Sequential(nn.Linear(16, self.scenario_len), nn.ReLU())

    def forward(self, x):
        # scenario = self.scenario1(scenario)
        # scenario = torch.mean(scenario,dim=0)
        # scenario = self.scenario3(self.scenario2(scenario))
        # input = torch.cat([x, scenario], dim=0)
        output = self.fcx2(self.fcx1(x))
        return output



# def test():
#     test_data = data['val_data']
#     E_model = torch.load('E_model.pt')
#     for i in range(len(test_data)):
#         with torch.no_grad():
#             x = torch.tensor(test_data[i]['x']).float()
#             scenario = torch.from_numpy(np.array(test_data[i]['scenario'])).float()
#             label = np.array(test_data[i]['obj_mean'])
#             prediction = np.array(E_model.forward(x, scenario))
#             error_ratio = (prediction-label)/label
#             print(error_ratio)
def get_data(problem_number, need_ind = False):
    p = Path(f'training_data/problem2_{problem_number}.pkl')
    f_save = open(p, 'rb')
    data = pkl.load(f_save)
    train_data = []
    train_data_ind = []  # use x-index version to reduce input size and see whether this works
    test_data = []
    test_data_ind = []
    train_label = []
    test_label = []
    train_size = 4500   #  4500train+500test
    if need_ind:
        for i in range(len(data)):
            if i < train_size:
                train_data_ind.append(data[i]['x_index'])
                train_label.append(data[i]['obj'])
            else:
                test_data_ind.append(data[i]['x_index'])
                test_label.append(data[i]['obj'])
        return torch.tensor(train_data_ind).float(), torch.tensor(train_label).float(), torch.tensor(test_data_ind).float(), torch.tensor(test_label).float()
    else:
        for i in range(len(data)):
            if i < train_size:
                train_data.append(data[i]['x'])
                train_label.append(data[i]['obj'])
            else:
                test_data.append(data[i]['x'])
                test_label.append(data[i]['obj'])
        return torch.tensor(train_data).float(), torch.tensor(train_label).float(), torch.tensor(test_data).float(), torch.tensor(test_label).float()
    
def train(problem_number, epoch=10000, batch_size = 32, need_ind = False):
    train_input, train_label, test_input, test_label = get_data(problem_number, need_ind)
    input_len = train_input.size()[1]
    model = Linear(input_len)
    Loss = torch.nn.MSELoss()
    if torch.cuda.is_available():
        model.cuda()
        Loss.cuda()
        train_input.cuda()
        train_label.cuda()
        test_input.cuda()
        test_label.cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

    for i in range(epoch):
        indecs = np.random.choice(len(train_input), batch_size, replace=False)
        input = train_input[indecs].cuda()
        label = train_label[indecs].cuda()
        optimizer.zero_grad()
        output = model.forward(input)
        # print(output)
        loss = Loss(label, output) / batch_size
        loss.backward()
        optimizer.step()
            # index = np.random.choice(1000)
            # try_x, try_scenario = x_and_its_scenario(index)
            # print(E_model.forward(torch.from_numpy(try_x).float(),torch.from_numpy((try_scenario)).float()))
            # print(labels[index])
        print(loss)
    if need_ind:
        torch.save(model,f'model/ind_model/ind_ model2_{problem_number}.pt')
    else:
        torch.save(model,f'model/model/model2_{problem_number}.pt')

if __name__=='__main__':
    batch_size = 32
    epoch = 10000
    need_ind = True
    for i in range(50):
        train(problem_number=i, epoch=epoch, batch_size=batch_size, need_ind=need_ind)
    # train(problem_number=14,epoch=epoch, batch_size=batch_size)