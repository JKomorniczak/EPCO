import os

default_n_threads = 1
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import numpy as np
from sklearn.datasets import make_classification
from problexity.classification import f1, f3, l2, n1, n3, n4, t1, clsCoef, hubs, t4
from Generate import GenComplexity
import matplotlib.pyplot as plt

np.random.seed(188)

def gen_pareto(gen, labels):
    aa = np.array(gen.measures_all)
            
    fig, axx = plt.subplots(len(gen.measures),len(gen.measures),figsize=(10,10))
    cols = plt.cm.coolwarm(np.linspace(0,1,gen.iters))
    
    for c1 in range(len(gen.measures)):
        for c2 in range(len(gen.measures)):
            try:
                ax = axx[c2,c1]
            except:
                ax = axx
            
            if c1==c2:
                ax.plot(aa[:,len(gen.measures),c1], c='k')
                ax.plot(aa[:,c1,c1], c='b', ls=':')
                if c1==0:
                    ax.set_ylabel(labels[c2])
                if c1==len(gen.measures)-1:
                    ax.set_xlabel(labels[c1])
            else:
                for iter in range(gen.iters):
                    ax.scatter(aa[iter,:,c1], aa[iter,:,c2], color=cols[iter], alpha=0.2, s=7)
                ax.scatter(aa[-1,:len(gen.measures)+1,c1],aa[-1,:len(gen.measures)+1,c2], c='b', marker='x', s=30)
                ax.scatter(0,0,c='k',marker='x')
                if c2==len(gen.measures)-1:
                    ax.set_xlabel(labels[c1])
                if c1==0:
                    ax.set_ylabel(labels[c2])

            ax.set_xticks([])
            ax.set_yticks([])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(ls=':')

    plt.tight_layout()
    plt.savefig('gen_pareto.png')
    plt.savefig('gen_pareto.pdf')

reps = 10
random_states = np.random.randint(100,10000,reps)

complexity_funs = [f1, f3, n3, clsCoef]
ranges = [
    [0.3, 0.9], #f1
    [0.6, 1.0], #f3
    [0.1, 0.6], #n3
    [0.4, 1.0], #clscoef
]
n_targets = 5

targets = []
for fun_id in range(len(complexity_funs)):
    t = np.linspace(ranges[fun_id][0], ranges[fun_id][1], n_targets)
    targets.append(t)   

targets = np.array(targets).swapaxes(0,1)
n_datasets = 12

n_samples=500
n_features=20

combined_datasets = np.zeros((reps, n_targets, n_datasets, n_samples, n_samples, n_features+1))
combined_results = np.zeros((reps, n_targets, n_datasets, 2, len(complexity_funs)))

# X_source, y_source = make_classification(n_samples=500, random_state=random_states[0])
X_source, y_source = make_classification(n_samples=200, random_state=random_states[0])
gen = GenComplexity(X_source, y_source, targets[-1], complexity_funs, vis=True)

gen.generate(iters=200, pop_size=70, cross_ratio=0.25, mut_ratio=0.1, decay=0.007)

# gen.gen_image()
labels=['F1', 'F3', 'N3', 'ClsCoef']
gen_pareto(gen, labels)