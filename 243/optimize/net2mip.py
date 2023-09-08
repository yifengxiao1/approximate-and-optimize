import torch
import gurobipy as gp
from gurobipy import GRB
from gurobipy import LinExpr
import torch.nn as nn
import pickle as pkl
import multiprocessing
import time

class Linear(nn.Module):
    def __init__(self, x_len):
        super(Linear, self).__init__()
        self.x_len = x_len
        self.fcx1 = nn.Sequential(nn.Linear(self.x_len, self.x_len), nn.ReLU())
        self.fcx2 = nn.Sequential(nn.Linear(self.x_len, 1), nn.ReLU())

    def forward(self, x):
        output = self.fcx2(self.fcx1(x))
        return output



def get_problem_and_net(problem_number):
    model = gp.read(f'../mps_data/problem2_{problem_number}.mps.mps')
    net = torch.load(f'../model/model2_{problem_number}.pt', map_location=torch.device('cpu'))
    return model, net

def generate_surrogate_problem(model, net, M_plus=1e5, M_minus=1e5):
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
                Z.append(model.addVar(vtype=gp.GRB.BINARY, name=z_name))
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

def solve_surrogate(problem, threads):
    model = gp.read(f'../surrogate_problem/surrogate2_{problem}.mps')
    model.setParam(GRB.Param.LogFile, f'../surrogate_problem/log/surrogate2_{problem}.log')
    model.setParam('Threads', threads)
    model.optimize()
    model.write(f'../surrogate_problem/sol/surrogate2_{problem}.sol')
    return



def solve_sub_problem(dic):

    sub_model = gp.Model()
    e_var = dic['e_var']
    y_var = dic['y_var']
    k_con = dic['k_cons']
    f_con = dic['flow_cons']
    if len(y_var) == 0:     # On some edges, there is no y varaiable, whcih means e_sport should be set to 0 and sub-problem optimal obj is 0
        return 0.0  
    if k_con['RHS'] == 0:
        return 0.0

    E, Y = {}, {}
    for ekey in e_var.keys():
        E[ekey]= sub_model.addVar(vtype=GRB.BINARY,obj=e_var[ekey], name = ekey)

    for yvar in y_var:
        Y[yvar] = sub_model.addVar(vtype=GRB.BINARY, name = yvar)


    temp = LinExpr()
    for term in k_con['linexp']:
        if 'y' in term:  
            temp.add(Y[term], 1.0)
    sub_model.addConstr(temp == k_con['RHS'])

    for con in f_con:
        i = 0
        temp = LinExpr()
        while i < len(con)-1:
            if 'y' in con[i+1]:
                if con[i] != '+':
                    temp.add(Y[con[i+1]], float(con[i]))
                else:
                    temp.add(Y[con[i+1]], 1.0)
            if 'e' in con[i+1]:
                temp.add(E[con[i+1]], float(con[i]))
            i += 1
        sub_model.addConstr(temp <= 0)
    sub_model.update()
    sub_model.setParam('Threads', 1)
    sub_model.optimize()
    
    return sub_model.objVal
    




def partition(problem, threads):
    model_sur = gp.read(f'../surrogate_problem/surrogate2_{problem}.mps')
    model_sur.setParam('Threads', threads)
    model_sur.setParam(GRB.Param.LogFile, f'../surrogate_problem/log/surrogate2_{problem}.log')
    model_sur.optimize()
    first_stage_obj = 0


    model = gp.read(f'../mps_data/problem2_{problem}.mps.mps')
    for var in model_sur.getVars():
        if 'x(' in str(var):
            if var.getAttr('x') == 1:
                model.getVarByName(var.getAttr('VarName')).setAttr('lb', 1)
                first_stage_obj += var.getAttr('Obj')
            elif var.getAttr('x') == 0:
                model.getVarByName(var.getAttr('VarName')).setAttr('ub', 0)
    model.update()

    
    vars = model.getVars()
    cons = model.getConstrs()
    edge = int(str(vars[-1]).split('_')[1].split('(')[1])  # Numbers start with 0

    sub_problem = {f'{i}':{'e_var':{}, 'y_var':[], 'k_cons':{'RHS':0}, 'flow_cons':[]} for i in range(edge+1)}
    
    for var in vars:
        if 'y' in str(var):
            temp =  str(var).split('_')[-2]
            sub_problem[temp]['y_var'].append(str(var).split(' ')[-1][:-1])

        if f'e_port' in str(var):
            temp = str(var).split('_')[1].split('(')[1]
            sub_problem[temp]['e_var'][str(var).split(' ')[-1][:-1]] = var.getAttr('Obj')

    for con in cons:
        if 'k' in str(con):
            temp = str(con).split('_')[-1][:-1]
            linexp = model.getRow(con)
            
            while 'x' in str(linexp.getVar(0)):
                if linexp.getVar(0).getAttr('lb') == 1:  
                    sub_problem[temp]['k_cons']['RHS'] = 1
                linexp.remove(linexp.getVar(0))
            
            sub_problem[temp]['k_cons']['linexp'] = str(linexp).split(' ')
                
        elif 'flow' in str(con):
            temp = str(con).split('_')[-2]
            linexp = model.getRow(con)
            sub_problem[temp]['flow_cons'].append(str(linexp).split(' '))

    return first_stage_obj, model_sur.objVal-first_stage_obj, sub_problem



    
    # pool.close()
    # pool.join()
    # return res_list

    





if __name__=='__main__':
    threads = 30
    results = {}
    for l in range(50):
        res_list = []
        T1 = time.perf_counter()             # Including time of reading model
        FSO, SSO_sur, sub_problem = partition(l,30)
        edge = len(sub_problem)
        pool = multiprocessing.Pool(processes=threads)
        for i in range((edge)//threads + 1):
            if (i+1)*threads <= edge:
                res = pool.map(solve_sub_problem,[sub_problem[f'{j}'] for j in range(i*threads, (i+1)*threads)])               
            else:
                inputs = [j for j in range(i*threads, edge)]
                if len(inputs) !=0 :
                    res = pool.map(solve_sub_problem, [sub_problem[f'{j}'] for j in inputs])
                    # res = pool.map(par, inputs)
            res_list.append(sum(res))
        pool.close()
        pool.join()
        second_stage_obj = sum(res_list)
        T2 = time.perf_counter()
        results[f'problem{l}']={'FSO':FSO, 'SSO':second_stage_obj, 'SSO_sur': SSO_sur, 'Time': T2-T1}
        print(f'problem{l} done-------------------------------------------')

    print(results)
    f_save = open('results.pkl', 'wb')
    pkl.dump(results, f_save)
    f_save.close()


    # sub_problem = partition(0,30)
    # print(sub_problem)
        

        
    
    
        