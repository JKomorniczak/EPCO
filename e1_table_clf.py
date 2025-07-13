import numpy as np
from problexity.classification import f1, f3, f4, l2, n1, n3, n4, t1, clsCoef, t4
from tabulate import tabulate

combined_results = np.load('res/combined_results.npy')
        
complexity_funs = [f1, f3, f4, l2, n1, n3, n4, t1, clsCoef, t4]

complexity_funs = [c.__name__.upper() for c in complexity_funs]
complexity_funs[8] = 'ClsCoef'
ranges = np.array([
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
])

targets = ['', 'easy', 'med-easy', 'medium', 'med-complex', 'complex']

print(ranges.shape) # 10 x 2
print(combined_results.shape)
# (10, 5, 11, 2, 10)
combined_results_mean = np.mean(combined_results, axis=0)
# (5, 11, 2, 10)
combined_results_mean = combined_results_mean[:,-1] #sum criterion
# (5, 2, 10)
print(combined_results_mean.shape)

rows = []
rows.append(targets)

for i in range(10):
    r = [complexity_funs[i]]
    r.extend(['%.3f (%.3f)' % (cr, cf) for cr, cf in zip(combined_results_mean[:,1,i], combined_results_mean[:,0,i])])
    rows.append(r)

print(tabulate(rows, tablefmt='latex'))