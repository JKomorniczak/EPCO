import numpy as np
from problexity.classification import f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4, t1, lsc, \
    density, clsCoef, hubs, t2, t3, t4, c1, c2
import matplotlib.pyplot as plt

results = np.load('res/e_individual_clf.npy')
print(results.shape) # 1 x 20 x 11 x 3
# score, 
# complexity,
# complexity diff from source

complexity_funs = [f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4, t1, lsc, density, clsCoef, hubs, t2, t3, t4]
complexity_funs = [c.__name__.upper() for c in complexity_funs]
complexity_funs[1] = 'F1v'
complexity_funs[14] = 'Density'
complexity_funs[15] = 'ClsCoef'
complexity_funs[16] = 'Hubs'
targets = np.linspace(0, 1, 11)

print(len(complexity_funs))

results =results[0] # 20 x 10 x 3
achieved_complexities = results[:,:,1]
diff_complexities = results[:,:,2]

fig, ax = plt.subplots(1,1,figsize=(10,3), sharex=True)

violin_parts = ax.violinplot(achieved_complexities.T)
ax.set_xticks(np.arange(len(complexity_funs))+1, complexity_funs, rotation=45)
ax.set_ylabel('Measure value')
ax.set_xlabel('Measure')
ax.grid(ls=':')

mask = [1,0,0,1,1,0,1,0,1,0,1,1,1,0,0,1,0,0,0,1]

for p_id, pc in enumerate(violin_parts['bodies']):
    pc.set_color(['b','r'][mask[p_id]])

violin_parts['cbars'].set_colors(np.array(['b','r'])[mask])
violin_parts['cmins'].set_colors(np.array(['b','r'])[mask])
violin_parts['cmaxes'].set_colors(np.array(['b','r'])[mask])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/single_criteria.png')
plt.savefig('figures/single_criteria.pdf')
plt.savefig('figures/single_criteria.eps')

