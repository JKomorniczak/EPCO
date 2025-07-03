import numpy as np
from problexity.classification import f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4, t1, lsc, \
    density, clsCoef, hubs, t2, t3, t4, c1, c2
import matplotlib.pyplot as plt

results = np.load('res/e_individual.npy')
print(results.shape) # 10 x 20 x 11 x 5

# clf, 
# clf diff from source, 
# score, 
# complexity,
# complexity diff from source

res_labels = ['Accuracy', 'Diff Accuracy', 'Gen score', 'Complexity', 'Diff Complexity']

complexity_funs = [f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4, t1, lsc, density, clsCoef, hubs, t2, t3, t4]
complexity_funs = [c.__name__ for c in complexity_funs]
targets = np.linspace(0, 1, 11)


# mean_res = np.mean(results, axis=0) # 20 x 11 x 5
mean_res = results[0]
achieved_complexities = mean_res[:,:,3]
diff_complexities = mean_res[:,:,4]


fig, ax = plt.subplots(1,1,figsize=(12,3), sharex=True)

violin_parts = ax.violinplot(achieved_complexities.T)
ax.set_xticks(np.arange(len(complexity_funs))+1, complexity_funs)
ax.grid(ls=':')

mask = [1,0,0,1,1,0,1,0,1,0,1,1,1,0,0,1,1,0,0,1]

for p_id, pc in enumerate(violin_parts['bodies']):
    pc.set_color(['b','r'][mask[p_id]])

violin_parts['cbars'].set_colors(np.array(['b','r'])[mask])
violin_parts['cmins'].set_colors(np.array(['b','r'])[mask])
violin_parts['cmaxes'].set_colors(np.array(['b','r'])[mask])
    
plt.tight_layout()
plt.savefig('foo.png')

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
