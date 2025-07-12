import os
from Generate import GenComplexity

default_n_threads = 1
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import numpy as np
from sklearn.datasets import make_classification
from problexity.classification import f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4, t1, lsc, \
    density, clsCoef, hubs, t2, t3, t4, c1, c2
np.random.seed(188)

reps = 1
random_states = np.random.randint(100,10000,reps)

complexity_funs = [f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4, t1, lsc, density, clsCoef, hubs, t2, t3, t4]
targets = np.linspace(0, 1, 10)

# prev_res = np.load('res/e_individual.npy')

results = np.zeros((reps, len(complexity_funs), len(targets), 3)) # score, 
                                                                  # complexity,
                                                                  # complexity diff from source
for rep_id, rs in enumerate(random_states):
    X_source, y_source = make_classification(n_samples=350, random_state=rs)

    for fun_id, fun in enumerate(complexity_funs):
        print('Measure: %s' % fun.__name__)
        c_source = fun(X_source, y_source)
        
        # if np.sum(prev_res[rep_id, fun_id]==0)<55:
        #     results[rep_id, fun_id] = prev_res[rep_id, fun_id]
        #     continue

        for target_id, target in enumerate(targets):
            gen = GenComplexity(X_source, y_source, [target], [fun])
            gen.generate(iters=50, pop_size=70, cross_ratio=0.25, mut_ratio=0.1)

            X, y = gen.return_best(0)
            
            results[rep_id, fun_id, target_id, 0] = gen.pop_scores[0,0]
                        
            c = fun(X,y)
            results[rep_id, fun_id, target_id, 1] = c
            results[rep_id, fun_id, target_id, 2] = c - c_source


        print(results[rep_id, fun_id])
        np.save('res/e_individual_clf.npy', results)
        
# f1, 0min
# f1v, 18s
# f2, 0min
# f3, 0min
# f4, 10s
# l1, 4s
# l2, 4s
# l3, 22s
# n1, 4min
# n2, 1min
# n3, 
# n4, 
# t1, 
# lsc, 
# density, 
# clsCoef, 
# hubs, 
# t2, 
# t3, 
# t4
