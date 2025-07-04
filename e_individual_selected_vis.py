import numpy as np
from problexity.classification import f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4, t1, lsc, \
    density, clsCoef, hubs, t2, t3, t4, c1, c2
import matplotlib.pyplot as plt

results = np.load('res/e_individual_selected.npy')
print(results.shape) # 10 x 20 x 11 x 5

# clf, 
# clf diff from source, 
# score, 
# complexity,
# complexity diff from source

res_labels = ['Accuracy', 'Diff Accuracy', 'Gen score', 'Complexity', 'Diff Complexity']

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
complexity_funs = [c.__name__ for c in complexity_funs]
targets = [0,1,2]

mean_res = np.mean(results, axis=0) # 20 x 11 x 5
# mean_res = results[0]

print(np.min(mean_res[:,:,1]))
print(np.max(mean_res[:,:,1]))

fig, ax = plt.subplots(1,5,figsize=(12,5), sharex=True, sharey=True)

for i in range(5):
    ax[i].set_title(res_labels[i])
    ax[i].imshow(mean_res[:,:,i], cmap='coolwarm', aspect='auto')
    ax[i].set_xticks(np.arange(len(targets))[::2], np.round(targets,1)[::2])

ax[0].set_yticks(np.arange(len(complexity_funs)), complexity_funs)
    
plt.tight_layout()
plt.savefig('foo.png')
