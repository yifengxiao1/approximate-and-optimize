import gurobipy as gp



# time = [60, 300, 600]
# for i in range(50):

m = gp.read('surrogate_problem/sol/surrogate2_0.sol')
print(m)