import numpy as np
from tabulate import tabulate

clf_complexity_funs = ['F1', 'F3', 'F4', 'L2', 'N1', 'N3', 'N4', 'T1', 'ClsCoef', 'T4']
clf_ranges = [
      [0.2, 0.9], #f1
    [0.45, 1.0], #f3
    [0.0, 0.85], #f4
    [0.05, 0.25], #l2
    [0.05, 0.3], #n1
    [0.1, 0.5], #n3
    [0.05, 0.3], #n4
    [0.6, 1.0], #t1
    [0.45, 1.0], #clscoef
    [0.5, 0.65]  #t4
]

reg_complexity_funs = ['C1', 'C2', 'S1', 'S2']
reg_ranges = [
    [0.9, 0.1], #c1
    [0.4, 0.0], #c2 
    [0.1, 0.25], #s1 
    [0.9, 1.0], #s2 
]
n_targets = 5

#clf
targets = []
for fun_id in range(len(clf_complexity_funs)):
    t = np.linspace(clf_ranges[fun_id][0], clf_ranges[fun_id][1], n_targets)
    targets.append(t)

targets_clf = np.array(targets).swapaxes(0,1)
# targets_clf.append([clf_complexity_funs])

print(tabulate(targets_clf, tablefmt='latex'))

#reg
targets = []
for fun_id in range(len(reg_complexity_funs)):
    t = np.linspace(reg_ranges[fun_id][0], reg_ranges[fun_id][1], n_targets)
    targets.append(t)

targets_reg = np.array(targets).swapaxes(0,1).astype(object)
print(reg_complexity_funs)
print(tabulate(targets_reg, tablefmt='latex'))

