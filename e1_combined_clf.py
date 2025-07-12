import os

default_n_threads = 1
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import numpy as np
from sklearn.datasets import make_classification
from problexity.classification import f1, f3, f4, l2, n1, n3, n4, t1, clsCoef, t4
from Generate import GenComplexity
np.random.seed(188)

reps = 10
random_states = np.random.randint(100,10000,reps)

complexity_funs = [f1, f3, f4, l2, n1, n3, n4, t1, clsCoef, t4]
ranges = [
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
n_targets = 5

targets = []
for fun_id in range(len(complexity_funs)):
    t = np.linspace(ranges[fun_id][0], ranges[fun_id][1], n_targets)
    targets.append(t)   

targets = np.array(targets).swapaxes(0,1)
n_datasets = len(complexity_funs)+1

n_samples=350
n_features=20

combined_datasets = np.zeros((reps, n_targets, n_datasets, n_samples, n_samples, n_features+1))
combined_results = np.zeros((reps, n_targets, n_datasets, 2, len(complexity_funs)))

for rep_id, rs in enumerate(random_states):
    X_source, y_source = make_classification(n_samples=n_samples, random_state=rs, n_features=n_features)

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
            
        np.save('res/combined_datasets.npy', combined_datasets)
        np.save('res/combined_results.npy', combined_results)
        