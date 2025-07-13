import numpy as np
from problexity.regression import c1, c2, c3, c4, s1, s2
from tabulate import tabulate

combined_results = np.load('res/combined_results_reg.npy')
        
complexity_funs = [c1, c2, s1, s2]
complexity_funs = [c.__name__.upper() for c in complexity_funs]
ranges = np.array([
    [0.9, 0.1], #c1
    [0.4, 0.0], #c2 
    [0.1, 0.25], #s1 
    [0.9, 1.0], #s2 
])

targets = ['', 'easy', 'med-easy', 'medium', 'med-complex', 'complex']


print(ranges.shape) # 11 x 2

print(combined_results.shape)
# (10, 5, 5, 2, 4)
combined_results_mean = np.mean(combined_results, axis=0)
# (5, 5, 2, 2)
combined_results_mean = combined_results_mean[:,-1] #sum criterion
# (5, 2, 4)
print(combined_results_mean.shape)

rows = []
rows.append(targets)

for i in range(4):
    r = [complexity_funs[i]]
    r.extend(['%.3f (%.3f)' % (cr, cf) for cr, cf in zip(combined_results_mean[:,1,i], combined_results_mean[:,0,i])])
    rows.append(r)

print(tabulate(rows, tablefmt='latex'))