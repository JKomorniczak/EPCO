import os

default_n_threads = 1
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import numpy as np
from sklearn.datasets import make_classification
from problexity.classification import f1, f3, l2, n1, n3, n4, t1, clsCoef, hubs, t4, f4
from Generate import GenComplexity
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

np.random.seed(188)

def gen_pareto(measures_all, measures, labels):
    fig, axx = plt.subplots(len(measures),len(measures),figsize=(8,8))
    cols = plt.cm.coolwarm(np.linspace(0,1,measures_all.shape[0]))
    
    for c1 in range(len(measures)):
        for c2 in range(len(measures)):
            try:
                ax = axx[c2,c1]
            except:
                ax = axx
            
            if c1==c2:
                ax.plot(gaussian_filter1d(measures_all[:,len(measures),c1],3), c='k')
                ax.plot(gaussian_filter1d(measures_all[:,c1,c1],3), c='b', ls=':')
                if c1==0:
                    ax.set_ylabel(labels[c2])
                if c1==len(measures)-1:
                    ax.set_xlabel(labels[c1])
            else:
                for iter in range(measures_all.shape[0]):
                    ax.scatter(measures_all[iter,:,c1], measures_all[iter,:,c2], color=cols[iter], alpha=0.15, s=10, lw=0)
                ax.scatter(measures_all[-1,:len(measures),c1],measures_all[-1,:len(measures),c2], c='b', marker='x', s=30)
                ax.scatter(measures_all[-1,len(measures),c1],measures_all[-1,len(measures),c2], c='k', marker='x', s=30)
                # ax.scatter(0,0,c='k',marker='x')
                if c2==len(measures)-1:
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


# complexity_funs = [f1, f3, f4, l2, n1, n3, n4, t1, clsCoef, hubs, t4]
# ranges = [
#     [0.3, 0.9], #f1
#     [0.6, 1.0], #f3
#     [0.3, 0.9], #f4
#     [0.05, 0.3], #l2
#     [0.05, 0.3], #n1
#     [0.1, 0.6], #n3
#     [0.1, 0.4], #n4
#     [0.6, 1.0], #t1
#     [0.4, 1.0], #clscoef
#     [0.9, 1.0], #hubs
#     [0.4, 0.6]  #t4
# ]


complexity_funs = [f1, f4, n1, t1, clsCoef]
ranges = [
    [0.3, 0.9], #f1
    # [0.6, 1.0], #f3
    [0.3, 0.9], #f4
    # [0.05, 0.3], #l2
    [0.05, 0.3], #n1
    # [0.1, 0.6], #n3
    # [0.1, 0.4], #n4
    [0.6, 1.0], #t1
    [0.4, 1.0], #clscoef
    # [0.9, 1.0], #hubs
    # [0.4, 0.6]  #t4
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


# GEN
X_source, y_source = make_classification(n_samples=200, random_state=random_states[0])
gen = GenComplexity(X_source, y_source, targets[-1], complexity_funs, vis=True)

gen.generate(iters=200, pop_size=100, cross_ratio=0.25, mut_ratio=0.1, decay=0.007)
np.save('res/gen_example_measures.npy', gen.measures_all)


# #DRAW
# measures_all = np.load('res/gen_example_measures.npy')

# # gen.gen_image()
# labels=['F1', 'F3', 'F4', 'L2', 'N1', 'N3', 'N4', 'T1', 'ClsCoef', 'Hubs', 'T4']
# gen_pareto(measures_all, complexity_funs, labels)