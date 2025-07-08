import numpy as np
from problexity.classification import f1, f3, f4, l2, n1, n3, n4, t1, clsCoef, hubs, t4
import matplotlib.pyplot as plt
from tabulate import tabulate

np.random.seed(188)

combined_results = np.load('res/combined_results.npy')
        
complexity_funs = [f1, f3, f4, l2, n1, n3, n4, t1, clsCoef, hubs, t4]
n_targets = 5
n_datasets = 12

complexity_funs = [c.__name__.upper() for c in complexity_funs]
complexity_funs[8] = 'ClsCoef'
complexity_funs[9] = 'Hubs'

targets = ['', 'easy', 'med-easy', 'medium', 'med-complex', 'complex']

ranges = np.array([
    [0.3, 0.9], #f1
    [0.6, 1.0], #f3
    [0.3, 0.9], #f4
    [0.05, 0.3], #l2
    [0.05, 0.3], #n1
    [0.1, 0.6], #n3
    [0.1, 0.4], #n4
    [0.6, 1.0], #t1
    [0.4, 1.0], #clscoef
    [0.9, 1.0], #hubs
    [0.4, 0.6]  #t4
])
print(ranges.shape) # 11 x 2

print(combined_results.shape)
# (10, 5, 12, 2, 11)
combined_results_mean = np.mean(combined_results, axis=0)
# (5, 12, 2, 11)
combined_results_mean = combined_results_mean[:,-1] #sum criterion
# (5, 2, 11)
print(combined_results_mean.shape)

rows = []
rows.append(targets)

for i in range(11):
    r = [complexity_funs[i]]
    r.extend(['%.3f (%.3f)' % (cr, cf) for cr, cf in zip(combined_results_mean[:,1,i], combined_results_mean[:,0,i])])
    rows.append(r)

print(tabulate(rows, tablefmt='latex'))