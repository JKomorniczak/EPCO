import numpy as np
from problexity.classification import f1, f3, f4, l2, n1, n3, n4, t1, clsCoef, hubs, t4
import matplotlib.pyplot as plt

np.random.seed(188)

combined_datasets = np.load('res/combined_datasets.npy')
combined_results = np.load('res/combined_results.npy')
        
complexity_funs = [f1, f3, f4, l2, n1, n3, n4, t1, clsCoef, hubs, t4]
n_targets = 5
n_datasets = 12

ranges = np.array([
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
])
print(ranges.shape) # 11 x 2

print(combined_datasets.shape)
print(combined_results.shape)
# (10, 5, 12, 500, 21)
# (10, 5, 12, 2, 11)

# combined_results_mean = np.mean(combined_results, axis=0)
combined_results_mean = combined_results[9]
# targets, datasets, (scores, complexities), measures

fig, ax = plt.subplots(2, n_targets, figsize=(15,7), sharex=True, sharey=True)
for t_id in range(n_targets):
    com_pct = combined_results_mean[t_id,:,1]
    print(com_pct.shape) # 12 x 11
    com_pct -= ranges[:,0]
    com_pct /= ranges[:,1]
     
    ax[0, t_id].imshow(combined_results_mean[t_id,:,0], cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
    ax[0, t_id].set_title('target: %i' % t_id)

    ax[1, t_id].imshow(com_pct, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)

    if t_id==0:
        ax[0, t_id].set_ylabel('fitness')
        ax[1, t_id].set_ylabel('complexity')
    
    ax[0, t_id].set_xticks(np.arange(11), [c.__name__ for c in complexity_funs], rotation=90)
    ax[1, t_id].set_xticks(np.arange(11), [c.__name__ for c in complexity_funs], rotation=90)

    ax[0, t_id].set_yticks(np.arange(12))
    ax[1, t_id].set_yticks(np.arange(12))

plt.tight_layout()
plt.savefig('foo.png')
    