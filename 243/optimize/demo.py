import gurobipy as gp
from pathlib import Path
import numpy as np
import pickle as pkl
from pathlib import Path
import numpy as np


problem = 0







# def solve_sub(model, edge):
#     with gp.Env() as env, gp.Model(env=env) as model:
#         # define model
#         sub_model = 
#         # retrieve data from model

# if __name__ == '__main__':
#     with mp.Pool() as pool:
#         pool.map
#         pool.map(solve_model, [input_data1, input_data2, input_data3]
import pandas
f = open('results.pkl', 'rb')
results = pkl.load(f)
f.close()

for i in range(50):

    results[f'problem{i}']['OBJ'] = results[f'problem{i}']['FSO']+results[f'problem{i}']['SSO']
    f = open(f'../reopt_p2/problem2_{i}.log', 'rb')
    GRB = str(f.readlines()[-1]).split(' ')[2]
    results[f'problem{i}']['GRB'] = GRB

p = pandas.DataFrame(results)
p.to_csv('results.csv')





# b=m.remove(a.getVar(0))
# m.update()
# a = m.getRow(cons[1000])
# print(a)