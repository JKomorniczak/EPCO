import os

# default_n_threads = 1
# os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
# os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
# os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import numpy as np
from sklearn.datasets import make_regression
from problexity.regression import c1, c2, c3, c4, l1, l2, s1, s2, s3, l3, s4, t2
from sklearn.model_selection import cross_val_score
from Generate import GenComplexity
from sklearn.linear_model import LinearRegression

np.random.seed(188)

reps = 10
random_states = np.random.randint(100,10000,reps)

complexity_funs = [c1, c2, c3, c4, l1, l2, s1, s2, s3, l3, s4, t2]
targets = np.linspace(0, 1, 6)

results = np.zeros((reps, len(complexity_funs), len(targets), 5)) # clf, 
                                                                  # clf diff from source, 
                                                                  # score, 
                                                                  # complexity,
                                                                  # complexity diff from source
for rep_id, rs in enumerate(random_states):
    X_source, y_source = make_regression(n_samples=100, random_state=rs)

    for fun_id, fun in enumerate(complexity_funs):
        print('Measure: %s' % fun.__name__)
        c_source = fun(X_source, y_source)

        for target_id, target in enumerate(targets):
            gen = GenComplexity(X_source, y_source, [target], [fun])
            gen.generate(iters=50, pop_size=70, cross_ratio=0.25, mut_ratio=0.1)

            X, y = gen.return_best(0)
            
            reg = LinearRegression()
            s1 = cross_val_score(reg, X_source, y_source, cv=10)
            s2 = cross_val_score(reg, X, y, cv=10)

            results[rep_id, fun_id, target_id, 0] = np.mean(s2)
            results[rep_id, fun_id, target_id, 1] = np.mean(s2) - np.mean(s1)
            
            results[rep_id, fun_id, target_id, 2] = gen.pop_scores[0,0]
                        
            c = fun(X,y)
            results[rep_id, fun_id, target_id, 3] = c
            results[rep_id, fun_id, target_id, 4] = c - c_source

        print(results[rep_id, fun_id])
        np.save('res/e_individual_reg_mini.npy', results)
    
    exit()