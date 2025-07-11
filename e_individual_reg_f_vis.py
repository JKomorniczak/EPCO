import numpy as np
from problexity.regression import c1, c2, c3, c4, l1, l2, s1, s2, s3, l3, s4, t2
import matplotlib.pyplot as plt

results_f = np.load('res/e_individual_reg_f.npy')
results_c4 = np.load('res/e_individual_reg_f_c4.npy')
print(results_f.shape) # 10 x 20 x 10 x 5
print(results_c4.shape) # 10 x 20 x 10 x 5

results = np.zeros((10, 12, 10, 5))
results[:,[0,1,2,4,5,6,7,8,9,10,11]] = results_f
results[:,3] = results_c4[:,0]
# clf, 
# clf diff from source, 
# score, 
# complexity,
# complexity diff from source

complexity_funs = [c1, c2, c3, c4, l1, l2, s1, s2, s3, l3, s4, t2]
complexity_funs = [c.__name__.upper() for c in complexity_funs]

# mean_res = np.mean(results, axis=0) # 20 x 11 x 5
mean_res = results[0]
achieved_complexities = mean_res[:,:,3]
diff_complexities = mean_res[:,:,4]

fig, ax = plt.subplots(1,1,figsize=(10,3), sharex=True)

violin_parts = ax.violinplot(achieved_complexities.T)
ax.set_xticks(np.arange(len(complexity_funs))+1, complexity_funs, rotation=45)
ax.set_ylabel('Measure value')
ax.set_xlabel('Measure')
ax.grid(ls=':')

ax.set_ylim(0,1)
    
mask = [1,1,1,1,0,0,1,1,1,0,0,0,0]

for p_id, pc in enumerate(violin_parts['bodies']):
    pc.set_color(['b','r'][mask[p_id]])

violin_parts['cbars'].set_colors(np.array(['b','r'])[mask])
violin_parts['cmins'].set_colors(np.array(['b','r'])[mask])
violin_parts['cmaxes'].set_colors(np.array(['b','r'])[mask])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/single_criteria_reg.png')
plt.savefig('figures/single_criteria_reg.pdf')
plt.savefig('figures/single_criteria_reg.eps')

