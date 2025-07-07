import os

default_n_threads = 1
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import numpy as np
from sklearn.datasets import make_classification
from problexity.classification import f1, f3, f4, l2, n1, n3, n4, t1, clsCoef, hubs, t4
from sklearn.model_selection import cross_val_score
from Generate import GenComplexity
from sklearn.naive_bayes import GaussianNB
np.random.seed(188)

reps = 10
random_states = np.random.randint(100,10000,reps)

complexity_funs = [f1, f3, f4, l2, n1, n3, n4, t1, clsCoef, hubs, t4]
ranges = [
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
]
n_targets = 3

results = np.zeros((reps, len(complexity_funs), n_targets, 5)) # clf, 
                                                                  # clf diff from source, 
                                                                  # score, 
                                                                  # complexity,
                                                                  # complexity diff from source

for rep_id, rs in enumerate(random_states):
    X_source, y_source = make_classification(n_samples=500, random_state=rs)

    for fun_id, fun in enumerate(complexity_funs):
        print('Measure: %s' % fun.__name__)
        c_source = fun(X_source, y_source)

        targets = np.linspace(ranges[fun_id][0], ranges[fun_id][1], n_targets)
        
        for target_id, target in enumerate(targets):
            gen = GenComplexity(X_source, y_source, [target], [fun])
            gen.generate(iters=50, pop_size=70, cross_ratio=0.25, mut_ratio=0.1)

            X, y = gen.return_best(0)
            
            clf = GaussianNB()
            s1 = cross_val_score(clf, X_source, y_source, cv=10)
            s2 = cross_val_score(clf, X, y, cv=10)

            results[rep_id, fun_id, target_id, 0] = np.mean(s2)
            results[rep_id, fun_id, target_id, 1] = np.mean(s2) - np.mean(s1)
            
            results[rep_id, fun_id, target_id, 2] = gen.pop_scores[0,0]
                        
            c = fun(X,y)
            results[rep_id, fun_id, target_id, 3] = c
            results[rep_id, fun_id, target_id, 4] = c - c_source


        print(results[rep_id, fun_id])
        np.save('res/e_individual_selected.npy', results)