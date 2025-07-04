import numpy as np
from problexity.classification import f1, f3, f4, l2, n1, n3, n4, t1, clsCoef, hubs, t4
import matplotlib.pyplot as plt

np.random.seed(188)

combined_datasets = np.load('res/combined_datasets.npy')
combined_results = np.load('res/combined_results.npy')
        
complexity_funs = [f1, f3, f4, l2, n1, n3, n4, t1, clsCoef, hubs, t4]
n_targets = 5
n_datasets = 12


# combined_datasets[rep_id, target_id, n, :, :n_features] = X
# combined_datasets[rep_id, target_id, n, :, -1] = y

# combined_results[rep_id, target_id, n, 0] = gen.pop_scores[n]
# combined_results[rep_id, target_id, n, 1] = [fun(X,y) for fun in gen.measures]

print(combined_datasets.shape)
print(combined_results.shape)
# (10, 5, 12, 500, 21)
# (10, 5, 12, 2, 11)

# combined_results_mean = np.mean(combined_results, axis=0)
combined_results_mean = combined_results[0]
# targets, datasets, (scores, complexities), measures

fig, ax = plt.subplots(2, n_targets, figsize=(15,7), sharex=True, sharey=True)
for t_id in range(n_targets):
    ax[0, t_id].imshow(combined_results_mean[t_id,:,0], cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
    ax[0, t_id].set_title('target: %i' % t_id)

    ax[1, t_id].imshow(combined_results_mean[t_id,:,1], cmap='coolwarm', aspect='auto', vmin=0, vmax=1)

    if t_id==0:
        ax[0, t_id].set_ylabel('fitness')
        ax[1, t_id].set_ylabel('complexity')
    
    ax[0, t_id].set_xticks(np.arange(11), [c.__name__ for c in complexity_funs], rotation=90)
    ax[1, t_id].set_xticks(np.arange(11), [c.__name__ for c in complexity_funs], rotation=90)

    ax[0, t_id].set_yticks(np.arange(12))
    ax[1, t_id].set_yticks(np.arange(12))

plt.tight_layout()
plt.savefig('foo.png')
    