import gurobipy as gp
from gurobipy import GRB
import pickle as pkl
import numpy as np
import multiprocessing

def get_problem(problem_number=0):
    m = gp.read(f'../mps_data/problem2_{problem_number}.mps.mps')
    return m

def get_req_path_dict(m):
    # dict = {request:[begining_path, end_path], ...}
    if m == None:
        print('No problem input')
        return
    else:
        no_request = 0
        no_path = 0
        req_path_dict = {0:[0]}
        model = m
        while model.getVarByName(f'x({no_request}_{no_path})') is not None:
            req_path_dict[no_request] = [no_path]
            while model.getVarByName(f'x({no_request}_{no_path})') is not None:          
                no_path += 1
            req_path_dict[no_request].append(no_path-1)
            no_request += 1
        return req_path_dict
            
def generate_get_req_path_dict():
    for i in range(50):
        model = get_problem(problem_number=i)
        dictionary = get_req_path_dict(model)
        f_save = open(f'../index_data/req_path_dict_2_{i}.pkl', 'wb')
        pkl.dump(dictionary, f_save)
        f_save.close() 
    return

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

def generate_problem_under_first_stage_solution(problem_number, x_index):
    m = get_problem(problem_number)
    f_save = open(f'../index_data/req_path_dict_2_{problem_number}.pkl', 'rb')
    req_path_dict = pkl.load(f_save)
    f_save.close()
    for request in req_path_dict.keys():
        for path in range(req_path_dict[request][0], req_path_dict[request][1]+1):
            if path == x_index[request]:
                m.getVarByName(f'x({request}_{path})').setAttr('lb', 1)
            else:
                m.getVarByName(f'x({request}_{path})').setAttr('ub', 0)
    m.update()
    return m

def get_x_and_obj(problem_number,batch_size):
    i = problem_number
    data = []
    time_limit = 300
    j=0
    while j < batch_size:
        sample = {}
        x, x_index = generate_first_stage_solution(i)
        m = generate_problem_under_first_stage_solution(i, x_index)
        m.setParam('Timelimit', time_limit)
        m.setParam('OutputFlag', 0)
        m.optimize()
        FSO = 0
        for var in m.getVars():
            if 'x' in str(var):
                FSO += var.getAttr('x')
        # if m.MIPGap != GRB.INFINITY:
        sample['x'] = x
        sample['x_index'] = x_index
        obj = m.objVal - FSO
        for var in m.getVars():
            if 'x' in str(var):
                obj -= var.getAttr('x') * var.getAttr('Obj')
        sample['obj'] = obj
        data.append(sample)
        j+=1
    return data

def parallel(size, process=1):
    datasize = size  
    for j in range(50):
        pool = multiprocessing.Pool(processes=process)
        results = []
        for i in range(process):
            result = pool.apply_async(get_x_and_obj, (j, int(datasize/process)))
            results.append(result)
        pool.close()
        pool.join()
        data = []
        for res in results:
            data += res.get()
        f_save = open(f'../training_data_modified/problem2_{j}.pkl', 'wb')
        pkl.dump(data, f_save)
        f_save.close()
        print(f'Problem {j} done')


            
if __name__ == '__main__':
    parallel(size=50000, process=50)


        



# model = get_problem()
# x = model.getVarByName('x(1_0)')
# req_path_dict = get_req_path_dict(model)
# model.write('2_0.lp')




# print(req_path_dict)
