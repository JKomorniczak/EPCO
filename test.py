import numpy as np
from sklearn.datasets import make_classification, make_blobs
from problexity.classification import f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4, t1, lsc, \
    density, clsCoef, hubs, t2, t3, t4, c1, c2
import matplotlib.pyplot as plt
from tqdm import tqdm

from Generate import GenComplexity

### target
complexity_fun = [f1, n3, t1, density]
targets = [0.3, 0.6, 0.6, 0.6]

X_source, y_source = make_classification(n_samples=200)

# optimize
mirror = GenComplexity(X_source, y_source, targets, complexity_fun, vis=True)
mirror.generate(iters=50, pop_size=70, cross_ratio=0.25, mut_ratio=0.1)

X, y = mirror.return_best(2)

mirror.gen_image()
mirror.gen_pareto()

