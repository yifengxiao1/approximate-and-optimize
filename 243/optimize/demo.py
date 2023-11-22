import gurobipy as gp
from pathlib import Path
import numpy as np
import pickle as pkl
from pathlib import Path
import numpy as np
import pandas
from data_generate import generate_first_stage_solution
import torch


print(generate_first_stage_solution(1))



# def solve_sub(model, edge):
#     with gp.Env() as env, gp.Model(env=env) as model:
#         # define model
#         sub_model = 
#         # retrieve data from model

# if __name__ == '__main__':
#     with mp.Pool() as pool:
#         pool.map
#         pool.map(solve_model, [input_data1, input_data2, input_data3]
def tocsv():
    import pandas
    f = open('results_modified.pkl', 'rb')
    results = pkl.load(f)
    f.close()
    for i in range(10):

        results[f'problem{i}']['OBJ'] = results[f'problem{i}']['FSO']+results[f'problem{i}']['SSO']
        f = open(f'../reopt_p2/problem2_{i}.log', 'rb')
        GRB = str(f.readlines()[-1]).split(' ')[2]
        results[f'problem{i}']['GRB'] = GRB

    p = pandas.DataFrame(results)
    p.to_csv('results_modified.csv')

# model = gp.read('../mps_data/problem2_0.mps.mps')
# model.read('0.sol')
# print(model.objVal)
# # model.update()
# vars = model.getVars()
# x = []
# SSO = 0
# for var in vars:
#     if 'x' in str(var):
#         x.append(var.getAttr('Start'))
#     if 'e' in str(var):
#         SSO += var.getAttr('x') * var.getAttr('Obj')

# tocsv()





# b=m.remove(a.getVar(0))
# m.update()
# a = m.getRow(cons[1000])
# print(a)
# for i in range(1,50):
#     model = gp.read(f'../surrogate_problem/surrogate2_{i}.mps')
#     for var in model.getVars():
#         if 'z' in str(var):
#             var.setAttr('VType', gp.GRB.CONTINUOUS)
#             var.setAttr('lb', 0)
#             var.setAttr('ub', 1)
#     model.update()
#     model.optimize()
#     model.write(f'demo/{i}.sol')
#     FSO = 0
#     for var in model.getVars():
#         if 'x' in str(var):
#             FSO += var.getAttr('obj')*var.getAttr('x')
#         if 'z' in str(var):
#             if var.getAttr('x')!=-0.0 or var.getAttr('x')!= 1.0:
#                 print(str(var))
#                 print(var.getAttr('x'))
#     print(model.objVal - FSO)
#     break

            


