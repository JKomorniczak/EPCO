import numpy as np
from problexity.regression import c1, c2, c3, c4, l1, l2, s1, s2, s3, l3, s4, t2
import matplotlib.pyplot as plt

results = np.load('res/e_individual_reg_mini.npy')
results2 = np.load('res/e_individual_reg_mini2.npy')

print(results.shape)
print(results2.shape)
# (10, 12, 6, 5)
# (10, 12, 6, 5)

results2[:,[0,1,2]] = results[:,[0,1,2]]
results = results2

print(results.shape) # 10 x 20 x 11 x 5

# clf, 
# clf diff from source, 
# score, 
# complexity,
# complexity diff from source

res_labels = ['Accuracy', 'Diff Accuracy', 'Gen score', 'Complexity', 'Diff Complexity']

complexity_funs = [c1, c2, c3, c4, l1, l2, s1, s2, s3, l3, s4, t2]
complexity_funs = [c.__name__.upper() for c in complexity_funs]


# mean_res = np.mean(results, axis=0) # 20 x 11 x 5
mean_res = results[0]
achieved_complexities = mean_res[:,:,3]
diff_complexities = mean_res[:,:,4]


fig, axx = plt.subplots(2,1,figsize=(10,6), sharex=True)

for ax in axx:
    violin_parts = ax.violinplot(achieved_complexities.T)
    ax.set_xticks(np.arange(len(complexity_funs))+1, complexity_funs, rotation=45)
    ax.set_ylabel('Measure value')
    ax.set_xlabel('Measure')
    ax.grid(ls=':')
        
    mask = [1,1,1,0,0,0,1,0,1,0,0,0]

    for p_id, pc in enumerate(violin_parts['bodies']):
        pc.set_color(['b','r'][mask[p_id]])

    violin_parts['cbars'].set_colors(np.array(['b','r'])[mask])
    violin_parts['cmins'].set_colors(np.array(['b','r'])[mask])
    violin_parts['cmaxes'].set_colors(np.array(['b','r'])[mask])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axx[0].set_ylim(0,1)
axx[1].set_ylim(2.4,2.8)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/single_criteria_reg.png')
plt.savefig('figures/single_criteria_reg.pdf')
plt.savefig('figures/single_criteria_reg.eps')

ranges = [
    [0.7, 0.1], #c1 5min || 7min
    [0.3, 0.0], #c2 4min || 7min
    [0.3, 0.9], #c3 50min || 47min
    # [0.05, 0.3], #c4 2,5h || 52min
    # [0.05, 0.3], #l1
    # [0.1, 0.6], #l2
    [0.05, 0.4], #s1
    # [0.6, 1.0], #s2
    [0.0, 0.2], #s3
    # [0.9, 1.0], #l3
    # [0.4, 0.6]  #s4
    # [0.4, 0.6]  #t2
]
