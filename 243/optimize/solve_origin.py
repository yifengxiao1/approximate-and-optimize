import gurobipy as gp
import multiprocessing
from gurobipy import GRB

def solve_origin(problem, threads):
    model = gp.read(f'../mps_data/problem2_{problem}.mps.mps')
    model.setParam(GRB.Param.LogFile, f'../reopt_p2/problem2_{problem}.log')
    model.setParam('TimeLimit', 600)
    model.setParam('Threads', threads)
    model.optimize()
    return

if __name__ == '__main__':
    for i in range(50):
        solve_origin(i, 30)