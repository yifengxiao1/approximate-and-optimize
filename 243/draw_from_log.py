import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

def draw(problem, AO):
    # if AO:
    #     file = f'log/problem2_{problem}.log'
    #     thread = 1
    # else:
    #     file = f'reopt_p2/problem2_{problem}.log'
    #     thread = 2
    file = f'reopt_p2/problem2_{problem}.log'
    lines = []
    Start = 'logging started Wed Sep'
    ready = 0
    record = False
    x = [] # obj
    y = [] # time
    with open(file) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if Start in line:
                record = True
            if 'Cutting planes:' in line:
                record = False
            if record:
                if 'Found heuristic solution: objective' in line:
                    data = line.split(' ')
                    data = [x for x in data if x!='']
                    x.append(float(line.split(' ')[-1][:-2]))
                    y.append(0)
                
                if ready:
                    data = line.split(' ')
                    data = [x for x in data if x!='']
                    if len(data) < 5:
                        pass
                    else:
                        x.append(float(data[-5]))
                        y.append(float(data[-1][:-2]))


                if 'Incumbent' in line:
                    ready = True


    return x, y



if __name__=='__main__':
    f = open('optimize/results.pkl', 'rb')
    results = pkl.load(f)
    f.close()
    for i in range(0, 50):
        objAO= results[f'problem{i}']['FSO'] + results[f'problem{i}']['SSO']
        timeAO = results[f'problem{i}']['Time']
        # objAO, timeAO = draw(i, True)
        objGu, timeGu = draw(i, False)
        plt.scatter(timeAO,np.log10(objAO),color = 'r')
        plt.plot(timeGu,np.log10(objGu),color = 'g')
        plt.xlabel("time(s)")
        plt.ylabel("log10(obj)")
        # plt.xlim(0,600)
        label = ('A&O', 'Gurobi')
        plt.legend(label, loc = "best")
        plt.savefig(f'pic/{i}.png')
        plt.close()


    

