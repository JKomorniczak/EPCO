import os

default_n_threads = 1
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import numpy as np
from sklearn.datasets import make_regression
from problexity.regression import c1, c2, c3, c4, s1, s2, s3
from Generate import GenComplexity
np.random.seed(188)

reps = 10
random_states = np.random.randint(100,10000,reps)

complexity_funs = [c1, c2, c3, c4, s1, s2]

ranges = [
    [0.9, 0.1], #c1 30s
    [0.55, 0.0], #c2 30s
    [0.0, 1.0], #c3 5min
    [0.0, 0.6], #c4 4min
    # [0.5, 0.9], #l1 2s
    # [0.05, 0.3], #l2 2s
    [0.1, 0.3], #s1 2min
    [0.6, 0.9], #s2 0
    # [0.0, 0.15], #s3 13min
    # [0.0, 0.15], #l3
    # [0.0, 0.15], #s4
    # [0.0, 0.15], #t2
]

n_targets = 5

targets = []
for fun_id in range(len(complexity_funs)):
    t = np.linspace(ranges[fun_id][0], ranges[fun_id][1], n_targets)
    targets.append(t)   

targets = np.array(targets).swapaxes(0,1)
n_datasets = len(complexity_funs)+1

n_samples=350
n_features=20

combined_datasets = np.zeros((reps, n_targets, n_datasets, n_samples, n_features+1))
combined_results = np.zeros((reps, n_targets, n_datasets, 2, len(complexity_funs)))

for rep_id, rs in enumerate(random_states):
    X_source, y_source = make_regression(n_samples=n_samples, random_state=rs, n_features=n_features)

    for target_id in range(n_targets):
        gen = GenComplexity(X_source, y_source, targets[target_id], complexity_funs)
        
        gen.generate(iters=150, pop_size=100,
                     cross_ratio=0.15, mut_ratio=0.05, 
                     decay=0.007)
        
        for n in range(n_datasets):
            X, y = gen.return_best(n)
            
            combined_datasets[rep_id, target_id, n, :, :n_features] = X
            combined_datasets[rep_id, target_id, n, :, -1] = y
            
            combined_results[rep_id, target_id, n, 0] = gen.pop_scores[n]
            combined_results[rep_id, target_id, n, 1] = [fun(X,y) for fun in gen.measures]
            
        np.save('res/combined_datasets_reg.npy', combined_datasets)
        np.save('res/combined_results_reg.npy', combined_results)
        