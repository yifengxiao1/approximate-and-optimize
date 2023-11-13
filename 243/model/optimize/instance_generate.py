import gurobipy as gp
from gurobipy import GRB
import pickle as pkl
import numpy as np
import multiprocessing
import numpy as np



def check_range(problem_number):
    model = gp.read(f'../mps_data/problem2_{problem_number}.mps.mps')
    xobj = []
    eobj = []
    for var in model.getVars():
        if 'x' in str(var):
            xobj.append(var.Obj)
        if 'e' in str(var):
            eobj.append(var.Obj)
    return xobj, eobj

def instance_generate(problem_number,generate_number):
    model = gp.read(f'../mps_data/problem2_{problem_number}.mps.mps')
    xobj, eobj = check_range(problem_number=problem_number)
    xmax = max(xobj)
    xmin = min(xobj)
    emax = max(eobj)
    emin = min(eobj)
    for i in range(generate_number):
        for var in model.getVars():
            if 'x' in str(var):
                var.setAttr('Obj',xmin + np.random.rand()*(xmax-xmin))
            if 'e' in str(var):
                var.setAttr('Obj',emin + np.random.rand()*(emax-emin))
        model.write(f'../mps_data_generated/problem2_{problem_number}_{i}.mps')
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

def generate_problem_under_first_stage_solution(problem_number,instance_number, x_index):
    m = gp.read(f'../mps_data_generated/problem2_{problem_number}_{instance_number}.mps')
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

def get_training_data(problem_number,instance_number,batch_size):
    i = problem_number
    data = []
    time_limit = 300
    j=0
    while j < batch_size:
        sample = {}
        x, x_index = generate_first_stage_solution(i)
        m = generate_problem_under_first_stage_solution(i,instance_number=instance_number,x_index=x_index)
        m.setParam('Timelimit', time_limit)
        m.setParam('OutputFlag', 0)
        m.optimize()
        # FSO = 0
        # for var in m.getVars():
        #     if 'x' in str(var):
        #         FSO += var.getAttr('x')
        # if m.status == GRB.Status.OPTIMAL:
        try:
            sample['x'] = x
            # sample['x_index'] = x_index
            obj = m.objVal
            xobj = []
            eobj = []
            for var in m.getVars():
                if 'x' in str(var):
                    xobj.append(var.Obj)
                if 'e' in str(var):
                    eobj.append(var.Obj)
            sample['obj'] = obj
            sample['xobj'] = xobj
            sample['eobj'] = eobj
            data.append(sample)
            j+=1
        except:
            pass
    return data



if __name__ == '__main__':
    for j in range(10):
        pool = multiprocessing.Pool(processes=50)
        results = []
        for i in range(50):
            result = pool.apply_async(get_training_data, (0, i, 200))
            results.append(result)
        pool.close()
        pool.join()
        data = []
        for res in results:
            data += res.get()
        f_save = open(f'../training_data_instance/problem2_0_{j+10}.pkl', 'wb')
        pkl.dump(data, f_save)
        f_save.close()
        print(f'Problem {j} done')