import net2mip as nm
import torch
import gurobipy as gp
from gurobipy import GRB
from gurobipy import LinExpr
import torch.nn as nn
import pickle as pkl
import multiprocessing
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Linear(nn.Module):
    def __init__(self, x_len):
        super(Linear, self).__init__()
        self.x_len = x_len
        self.fcx1 = nn.Sequential(nn.Linear(self.x_len, self.x_len), nn.ReLU())
        self.fcx2 = nn.Sequential(nn.Linear(self.x_len, 1), nn.ReLU())

    def forward(self, x):
        output = self.fcx2(self.fcx1(x))
        return output
    
    def f1(self, x):
        output = (self.fcx1(x))
        return output

# def findz(model, input):
#     x = torch.rand([2, 3, 224, 224])
#     for i in range(len(model)):
#     x = model[i](x)
#     if i == 2:
#         ReLu_out = x

def generate_first_stage_solution(problem_number):
    # x = [0,0,1,0,.....,0],  x_index = [5, 16, 58, 105, ...]
    f_save = open(f'../index_data/req_path_dict_2_{problem_number}.pkl', 'rb')
    req_path_dict = pkl.load(f_save)
    f_save.close()
    last_key = list(req_path_dict.keys())[-1]
    x_size = req_path_dict[last_key][-1] + 1
    x_index_size = len(req_path_dict.keys())
    x = [0 for _ in range(x_size)]
    x_index = [0 for _ in range(x_index_size)]
    for i, key in enumerate(req_path_dict.keys()):
        path = np.random.randint(low=req_path_dict[key][0], high=req_path_dict[key][1]+1)
        x[path] = 1
        x_index[i] = path
    return x, x_index



def generate_surrogate_problem_under_x(model, input, net, M_plus=1e5, M_minus=1e5):
    #  Firstly, delate all second stage vars and constrs
    vars = model.getVars()
    cons = model.getConstrs()
    for var in vars:
        if 'x' not in str(var):
            model.remove(var)
    for con in cons:
        if 'K' not in str(con):
            model.remove(con)
    model.update()
    nVar = len(model.getVars())
    #   Then add network to mip
    
    W, B = [], []
    for name, param in net.named_parameters():
        if 'weight' in name:
            W.append(param.detach().numpy())
        if 'bias' in name:
            B.append(param.detach().numpy())
    
    XX = []
    for k, (wt, b) in enumerate(zip(W, B)):
        outSz, inpSz = wt.shape
        X, S, Z = [], [], []
        for j in range(outSz):
            x_name = f'x_{k + 1}_{j}'
            s_name = f's_{k + 1}_{j}'
            z_name = f'z_{k + 1}_{j}'

            if k < len(W) - 1:
                X.append(model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=x_name))
                S.append(model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=s_name))
                Z.append(model.addVar(vtype=gp.GRB.INTEGER, lb=0, ub=1, name=z_name))
            else:
                X.append(model.addVar(vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name=x_name))
            
            _eq = 0
            for i in range(inpSz):
                # First layer weights are partially multiplied by gp.var and features
                if k == 0:
                    # Multiply gp vars
                    # if i < nVar:
                    _eq += wt[j][i] * model.getVars()[i]
                    # else:
                    #     _eq += wt[j][i] * scenario[i - nVar]
                else:
                    _eq += wt[j][i] * XX[-1][i]

            # Add bias
            _eq += b[j]

            # Add constraint for each output neuron
            if k < len(W) - 1:
                model.addConstr(_eq == X[-1] - S[-1], name=f"mult_{x_name}__{s_name}")
                model.addConstr(X[-1] <= M_plus * (1 - Z[-1]), name=f"bm_{x_name}")
                model.addConstr(S[-1] <= M_minus * (Z[-1]), name=f"bm_{s_name}")

            else:
                model.addConstr(_eq == X[-1], name=f"mult_out_{x_name}__{s_name}")

            # Save current layers gurobi vars
            XX.append(X)
    Q_var = XX[-1][-1]
    Q_var.setAttr('obj', 1)
    model.update()
    return model


def iteration(net, problem_number, x):
        if x==None:
            x, x_index = generate_first_stage_solution(problem_number)
        
        x = torch.tensor(x, dtype=torch.float)
        z = torch.sign(net.f1(x))
        print(z)
        m = gp.read(f'../surrogate_problem/surrogate2_{problem_number}.mps')
        for var in m.getVars():
            if 'z_' in str(var):
                if z[int((str(var).split('_')[-1].split('>')[0].split(' ')[0]))] == 0:
                    var.setAttr('ub',1)
                    var.setAttr('lb',1)    #  上边生成问题的代码是z=0激活，z=1不激活
                else:
                    var.setAttr('lb',0)
                    var.setAttr('ub',0)
        m.update()
        # m.write('ok.lp')
        m.optimize()
        out = []
        for var in m.getVars():
            if 'x(' in str(var):
                out.append(var.getAttr('x'))
        return out
        # for var in m.getVars():
        #     if 'z' in str(var):
        #         print(var)
        # out = []
        # for i in range(2):
        #     x = a[i](x)
        #     out.append(x)
        # print(out)

def find_best_solution(solutionset, problem_number):
    model = gp.read(f'../mps_data/problem2_{problem_number}.mps.mps')
    objs= []
    for sol in solutionset:
        for var in model.getVars():
            if 'x' in str(var):
                if sol[int(str(var).split('_')[-1][:-2].split(')')[0])] == 0:
                    var.setAttr('ub',0)
                    var.setAttr('lb',0)
                else:
                    var.setAttr('lb',1)
                    var.setAttr('ub',1)
        model.update()
        model.optimize()
        objs.append(model.objVal)
    return objs
                




if __name__=='__main__':
    T = 1
    objs = []
    for i in range(1,11):
        net = torch.load(f'../model/model2_{i}.pt', map_location=torch.device('cpu'))
        # m = gp.read(f'../surrogate_problem/surrogate2_{i}.mps')
        x = None
        solutionset = []
        for t in range(T):
            x = iteration(net, i,x)
            solutionset.append(x)
            break
        obj = find_best_solution(solutionset, i)
        objs.append(obj)
        # it = range(1,21)
        # plt.plot(it,objs,color='deepskyblue')
        # plt.xlabel("Iterations")
        # plt.ylabel("Obj")
        # p=pd.read_csv('results.csv')
        # grb = eval(p.loc[5,f'problem{i}'])
        # GRB = [grb for i in range(T)]
        # plt.plot(it, GRB,color = 'red')
        # # plt.legend()
        # plt.savefig(f'../iterpic/{i}.png')
        # plt.cla()
    print(objs)
        
        
    