import os

default_n_threads = 1
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import numpy as np
from sklearn.datasets import make_regression
from problexity.regression import c1, c2, s1, s3, c3, c4, s1, s2
from Generate import GenComplexity
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

np.random.seed(188)

def gen_pareto(measures_all, measures, labels):
    
    idx = np.random.choice(measures_all.shape[1], replace=False, size=40)
    idx = np.sort(idx)
    idx[:len(measures)+1] = np.arange(len(measures)+1)
    measures_all = measures_all[:70, idx]
    fig, axx = plt.subplots(len(measures),len(measures),figsize=(10,10))
    cols = plt.cm.coolwarm(np.linspace(0,1,measures_all.shape[0]))
    
    for c1 in range(len(measures)):
        for c2 in range(len(measures)):
            try:
                ax = axx[c2,c1]
            except:
                ax = axx
            
            if c1==c2:
                for aa in range(len(measures)):
                    ax.plot(gaussian_filter1d(measures_all[:,aa,c1],3), c='b', lw=0.25, label=labels[aa], ls=':')
                ax.plot(gaussian_filter1d(measures_all[:,len(measures),c1],3), c='k', label='$\Sigma$')
                ax.plot(gaussian_filter1d(measures_all[:,c1,c1],3), c='b', label=labels[c1])
                if c1==0:
                    ax.set_ylabel(labels[c2])
                if c1==len(measures)-1:
                    ax.set_xlabel(labels[c1])
            else:
                for iter in range(measures_all.shape[0]):
                    ax.scatter(measures_all[iter,:,c1], measures_all[iter,:,c2], color=cols[iter], alpha=0.5, s=10, lw=0)
                ax.scatter(measures_all[-1,:len(measures),c1],measures_all[-1,:len(measures),c2], c='b', marker='x', s=30)
                ax.scatter(measures_all[-1,len(measures),c1],measures_all[-1,len(measures),c2], c='k', marker='x', s=30)
                ax.scatter(0,0,c='k',marker='*')
                if c2==len(measures)-1:
                    ax.set_xlabel(labels[c1])
                if c1==0:
                    ax.set_ylabel(labels[c2])

            ax.set_xticks([])
            ax.set_yticks([])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(ls=':')
            
    line1 = Line2D([0], [0], label='criterion over evolution', color='b')
    line2 = Line2D([0], [0], label='mean criteria', color='k')
    # line3 = Line2D([0], [0], label='leaders', color='b', lw=0.25)
    point1 = Line2D([0], [0], label='individuals over evolution', markersize=5, 
                   markeredgecolor=cols[iter], marker='o', markerfacecolor=cols[iter],
                   linestyle='')
    point2 = Line2D([0], [0], label='leaders', markersize=5, 
                   markeredgecolor='b', marker='x', markerfacecolor='b',
                   linestyle='')

    plt.legend(bbox_to_anchor=(0.89, 0.045), loc="lower right", handles=[line1, line2, point1, point2],
                bbox_transform=fig.transFigure, ncol=5, frameon=False)

    plt.tight_layout()
    plt.savefig('figures/gen_pareto_r.png')
    plt.savefig('figures/gen_pareto_r.pdf')
    plt.savefig('figures/gen_pareto_r.eps')
    plt.savefig('foo.png')

random_states = np.random.randint(100,10000,10)
complexity_funs = [c1, c2, s1, s2]
ranges = [
    [0.9, 0.1], #c1
    [0.4, 0.0], #c2 
    # [0.0, 0.9], #c3 
    # [0.0, 0.5], #c4
    [0.1, 0.25], #s1 
    [0.9, 1.0], #s2 
]

n_targets = 5

targets = []
for fun_id in range(len(complexity_funs)):
    t = np.linspace(ranges[fun_id][0], ranges[fun_id][1], n_targets)
    targets.append(t)   

targets = np.array(targets).swapaxes(0,1)
n_datasets = 5
n_features=20

# GEN
# X_source, y_source = make_regression(n_samples=200, random_state=random_states[0], n_features=20, noise=1.0)
# gen = GenComplexity(X_source, y_source, targets[-1], complexity_funs, vis=True)

# gen.generate(iters=100, pop_size=100, cross_ratio=0.25, mut_ratio=0.1, decay=0.01)
# np.save('res/gen_example_measures_reg.npy', gen.measures_all)


# # #DRAW
measures_all = np.load('res/gen_example_measures_reg.npy')

# gen.gen_image()
labels=['C1', 'C2', 'S1', 'S2']
gen_pareto(measures_all, complexity_funs, labels)