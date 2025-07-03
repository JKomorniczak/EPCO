import numpy as np
from sklearn.datasets import make_classification
from problexity.classification import f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4, t1, lsc, \
    density, clsCoef, hubs, t2, t3, t4, c1, c2
from sklearn.model_selection import cross_val_score
from Generate import GenComplexity
from sklearn.naive_bayes import GaussianNB

np.random.seed(188)

reps = 10
random_states = np.random.randint(100,10000,reps)

complexity_funs = [f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4, t1, lsc, density, clsCoef, hubs, t2, t3, t4]
targets = np.linspace(0, 1, 11)

results = np.zeros((reps, len(complexity_funs), len(targets), 5)) # clf, 
                                                                  # clf diff from source, 
                                                                  # score, 
                                                                  # complexity,
                                                                  # complexity diff from source

for rep_id, rs in enumerate(random_states):
    X_source, y_source = make_classification(n_samples=500, random_state=rs)

    for fun_id, fun in enumerate(complexity_funs):
        print('Measure: %s' % fun.__name__)
        c_source = fun(X_source, y_source)

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
        np.save('res/e_individual.npy', results)