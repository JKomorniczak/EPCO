import os

default_n_threads = 1
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import numpy as np
from sklearn.datasets import make_regression
from problexity.regression import c1, c2, c3, s1, s3
from Generate import GenComplexity
np.random.seed(188)

reps = 10
random_states = np.random.randint(100,10000,reps)

complexity_funs = [c1, c2, c3, s1, s3]

ranges = [
    [0.7, 0.1], #c1 5min || 7min
    [0.3, 0.0], #c2 4min || 7min
    [0.5, 0.9], #c3 50min || 47min
    [0.05, 0.3], #s1
    [0.0, 0.15], #s3
]

n_targets = 5

targets = []
for fun_id in range(len(complexity_funs)):
    t = np.linspace(ranges[fun_id][0], ranges[fun_id][1], n_targets)
    targets.append(t)   

targets = np.array(targets).swapaxes(0,1)
n_datasets = len(complexity_funs)+1

n_samples=500
n_features=20

combined_datasets = np.zeros((reps, n_targets, n_datasets, n_samples, n_samples, n_features+1))
combined_results = np.zeros((reps, n_targets, n_datasets, 2, len(complexity_funs)))

for rep_id, rs in enumerate(random_states):
    X_source, y_source = make_regression(n_samples=500, random_state=rs, n_features=n_features)

    for target_id in range(n_targets):
        gen = GenComplexity(X_source, y_source, targets[target_id], complexity_funs)
        
        gen.generate(iters=200, pop_size=200,
                     cross_ratio=0.25, mut_ratio=0.1, 
                     decay=0.007)
        
        for n in range(n_datasets):
            X, y = gen.return_best(n)
            
            combined_datasets[rep_id, target_id, n, :, :n_features] = X
            combined_datasets[rep_id, target_id, n, :, -1] = y
            
            combined_results[rep_id, target_id, n, 0] = gen.pop_scores[n]
            combined_results[rep_id, target_id, n, 1] = [fun(X,y) for fun in gen.measures]
            
        np.save('res/combined_datasets_reg.npy', combined_datasets)
        np.save('res/combined_results_reg.npy', combined_results)
        