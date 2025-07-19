import os

default_n_threads = 1
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import numpy as np
from sklearn.datasets import make_classification
from problexity.classification import f1, f3, l2, n1, n3, n4, t1, clsCoef, hubs, t4, f4
from EPCO import EPCO
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

np.random.seed(188)

def gen_pareto(measures_all, measures, labels):
    
    idx = np.random.choice(measures_all.shape[1], replace=False, size=40)
    idx = np.sort(idx)
    idx[:len(measures)+1] = np.arange(len(measures)+1)
    measures_all = measures_all[:, idx]
    fig, ax = plt.subplots(1,4,figsize=(10,3))
    cols = plt.cm.coolwarm(np.linspace(0,1,measures_all.shape[0]))
    
    c1 = 0
    c2 = 1
    # print(measures_all[-1,:len(measures),c2])
    # print(measures_all[-1,len(measures),c2])
    # print(measures_all[-1,:10,c2])
    # exit()
    for iter in range(measures_all.shape[0]):
        ax[0].scatter(measures_all[iter,:,c1], measures_all[iter,:,c2], color=cols[iter], alpha=0.5, s=10, lw=0)
    ax[0].scatter(measures_all[-1,:len(measures),c1],measures_all[-1,:len(measures),c2], c='b', marker='x', s=30)
    # ax[0].scatter(measures_all[-1,len(measures),c1],measures_all[-1,len(measures),c2], c='k', marker='x', s=30)
    ax[0].scatter(0,0,c='k',marker='*')
    ax[0].set_xlabel(labels[c1])
    ax[0].set_ylabel(labels[c2])
    
    c1 = 0
    c2 = 2
    for iter in range(measures_all.shape[0]):
        ax[1].scatter(measures_all[iter,:,c1], measures_all[iter,:,c2], color=cols[iter], alpha=0.5, s=10, lw=0)
    ax[1].scatter(measures_all[-1,:len(measures),c1],measures_all[-1,:len(measures),c2], c='b', marker='x', s=30)
    # ax[1].scatter(measures_all[-1,len(measures),c1],measures_all[-1,len(measures),c2], c='k', marker='x', s=30)
    ax[1].scatter(0,0,c='k',marker='*')
    ax[1].set_xlabel(labels[c1])
    ax[1].set_ylabel(labels[c2])
    
    c1 = 1
    c2 = 2
    for iter in range(measures_all.shape[0]):
        ax[2].scatter(measures_all[iter,:,c1], measures_all[iter,:,c2], color=cols[iter], alpha=0.5, s=10, lw=0)
    ax[2].scatter(measures_all[-1,:len(measures),c1],measures_all[-1,:len(measures),c2], c='b', marker='x', s=30)
    # ax[2].scatter(measures_all[-1,len(measures),c1],measures_all[-1,len(measures),c2], c='k', marker='x', s=30)
    ax[2].scatter(0,0,c='k',marker='*')
    ax[2].set_xlabel(labels[c1])
    ax[2].set_ylabel(labels[c2])
            
    # for aa in range(len(measures)):
    #     ax[3].plot(gaussian_filter1d(measures_all[:,aa,c1],3), c='b', lw=0.25, label=labels[aa], ls=':')
    
    ccols = plt.cm.coolwarm([0.0, 0.5, 1.0])
    ccols[1,:3] -= 0.3
    ccols = np.clip(ccols, 0,1)
    
    c1 = 0
    ax[3].plot(gaussian_filter1d(measures_all[:,len(measures),c1],3), c=ccols[0])
    ax[3].plot(gaussian_filter1d(measures_all[:,c1,c1],3), c=ccols[0], ls=':')
    
    c1 = 1
    ax[3].plot(gaussian_filter1d(measures_all[:,len(measures),c1],3), c=ccols[1])
    ax[3].plot(gaussian_filter1d(measures_all[:,c1,c1],3), c=ccols[1], ls=':')
  
    c1 = 2
    ax[3].plot(gaussian_filter1d(measures_all[:,len(measures),c1],3), c=ccols[2])
    ax[3].plot(gaussian_filter1d(measures_all[:,c1,c1],3), c=ccols[2], ls=':')
  
    for aa in ax:
        aa.spines['top'].set_visible(False)
        aa.spines['right'].set_visible(False)
        aa.grid(ls=':')
            
    line1f = Line2D([0], [0], label='leader (F1)', color=ccols[0], ls=':')
    line2f = Line2D([0], [0], label='sum criteria (F1)', color=ccols[0])
    
    line1n = Line2D([0], [0], label='leader (N1)',color=ccols[1], ls=':')
    line2n = Line2D([0], [0], label='sum criteria (N1)', color=ccols[1])
    
    line1c = Line2D([0], [0], label='leader (ClsCoef)', color=ccols[2], ls=':')
    line2c = Line2D([0], [0], label='sum criteria (ClsCoef)', color=ccols[2])
    
    point1 = Line2D([0], [0], label='individuals\' scores', markersize=5, 
                   markeredgecolor=cols[iter], marker='o', markerfacecolor=cols[iter],
                   linestyle='')
    point2 = Line2D([0], [0], label='leaders\' scores', markersize=5, 
                   markeredgecolor='b', marker='x', markerfacecolor='b',
                   linestyle='')
    
    plt.subplots_adjust(top=0.9, bottom=0.35, wspace = 0.35)

    plt.legend(bbox_to_anchor=(0.89, 0.001), loc="lower right", handles=[point1, point2, line1f, line2f, line1n, line2n, line1c, line2c],
                bbox_transform=fig.transFigure, ncol=4, frameon=False)

    # plt.tight_layout()
    plt.savefig('figures/gen_pareto_mini.png')
    plt.savefig('figures/gen_pareto_mini.pdf')
    plt.savefig('figures/gen_pareto_mini.eps')
    plt.savefig('foo.png')

reps = 10
random_states = np.random.randint(100,10000,reps)

complexity_funs = [f1, n1, clsCoef]
ranges = [
    [0.2, 0.9], #f1
    # [0.45, 1.0], #f3
    # [0.0, 0.85], #f4
    # [0.05, 0.25], #l2
    [0.05, 0.3], #n1
    # [0.1, 0.5], #n3
    # [0.05, 0.3], #n4
    # [0.6, 1.0], #t1
    [0.45, 1.0], #clscoef
    # [0.5, 0.65]  #t4
]
n_targets = 5

targets = []
for fun_id in range(len(complexity_funs)):
    t = np.linspace(ranges[fun_id][0], ranges[fun_id][1], n_targets)
    targets.append(t)   

targets = np.array(targets).swapaxes(0,1)

# GEN
# X_source, y_source = make_classification(n_samples=200, random_state=random_states[0])
# gen = EPCO(X_source, y_source, targets[-1], complexity_funs, vis=True)

# gen.generate(iters=200, pop_size=200, cross_ratio=0.3, mut_ratio=0.1, decay=0.007)
# np.save('res/gen_example_measures_mini.npy', gen.measures_all)


# #DRAW
measures_all = np.load('res/gen_example_measures_mini.npy')

# gen.gen_image()
labels=['F1', 'N1', 'ClsCoef']
gen_pareto(measures_all, complexity_funs, labels)