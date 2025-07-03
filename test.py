import numpy as np
from sklearn.datasets import make_classification
from problexity.classification import f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4, t1, lsc, \
    density, clsCoef, hubs, t2, t3, t4, c1, c2
from sklearn.model_selection import cross_val_score
from Generate import GenComplexity
from sklearn.naive_bayes import GaussianNB

### target
complexity_fun = [f1]
targets = [0.3, 0.6, 0.6, 0.6]

X_source, y_source = make_classification(n_samples=200)

# optimize
gen = GenComplexity(X_source, y_source, targets, complexity_fun, vis=True)
gen.generate(iters=50, pop_size=70, cross_ratio=0.25, mut_ratio=0.1)

X, y = gen.return_best()

gen.gen_image()
gen.gen_pareto()

print(gen.target_complexity)

clf = GaussianNB()
s1 = cross_val_score(clf, X_source, y_source, cv=10)
s2 = cross_val_score(clf, X, y, cv=10)

print(np.mean(s1), np.std(s1))
print(np.mean(s2), np.std(s2))

