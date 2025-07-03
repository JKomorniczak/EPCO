import numpy as np
from problexity.classification import f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4, t1, lsc, \
    density, clsCoef, hubs, t2, t3, t4, c1, c2
import matplotlib.pyplot as plt

results = np.load('res/e_mir_individual.npy')
print(results.shape) # 10 x 20 x 5

# clf, 
# clf diff from source,
# score,
# complexity,
# complexity diff from source

res_labels = ['Accuracy', 'Diff Accuracy', 'Gen score', 'Complexity', 'Diff Complexity']

complexity_funs = [f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4, t1, lsc, density, clsCoef, hubs, t2, t3, t4]
complexity_funs = [c.__name__ for c in complexity_funs]
targets = np.linspace(0, 1, 11)

fig, ax = plt.subplots(5,1,figsize=(10,10), sharex=True)

cols = plt.cm.turbo(np.linspace(0,1,10))

for i in range(5):
    ax[i].set_title(res_labels[i])
    for r in range(10):
        ax[i].scatter(np.arange(20),results[r,:,i].T, color=cols[r])

ax[-1].set_xticks(np.arange(len(complexity_funs)), complexity_funs)

for aa in ax:
    aa.grid(ls=':')
plt.tight_layout()
plt.savefig('foo.png')
