import torch.nn as nn

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
