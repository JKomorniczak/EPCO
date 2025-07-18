
import numpy as np
from problexity.classification import f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4, t1, lsc, density, clsCoef, hubs, t2, t3, t4
from problexity.regression import c1, c2, c3, c4, l1, l2, s1, s2, s3, l3, s4, t2
import matplotlib.pyplot as plt

# clf
results_clf = np.load('res/e_individual_clf.npy')
results_clf = results_clf[0, :, :, 1]
print(results_clf.shape) # 20 x 10

complexity_funs_clf = [f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4, t1, lsc, density, clsCoef, hubs, t2, t3, t4]
complexity_funs_clf = [c.__name__.upper() for c in complexity_funs_clf]
complexity_funs_clf[1] = 'F1v'
complexity_funs_clf[14] = 'Density'
complexity_funs_clf[15] = 'ClsCoef'
complexity_funs_clf[16] = 'Hubs'


#reg
results_reg = np.load('res/e_individual_reg.npy')
results_reg = results_reg[0, :, :, 1]
print(results_reg.shape) # 12 x 10

complexity_funs_reg = [c1, c2, c3, c4, l1, l2, s1, s2, s3, l3, s4, t2]
complexity_funs_reg = [c.__name__.upper() for c in complexity_funs_reg]

mask_clf = [1,0,0,1,1,0,1,0,1,0,1,1,1,0,0,1,0,0,0,1]
mask_reg = [1,1,0,0,0,0,1,1,0,0,0,0,0]

fig, ax = plt.subplots(2,1,figsize=(10,5), sharey=True)

violin_parts = ax[0].violinplot(results_clf.T)
ax[0].set_xticks(np.arange(1,21), complexity_funs_clf, rotation=45)
ax[0].set_ylabel('classification complexity')

for p_id, pc in enumerate(violin_parts['bodies']):
    pc.set_color(['r','b'][mask_clf[p_id]])
violin_parts['cbars'].set_colors(np.array(['r','b'])[mask_clf])
violin_parts['cmins'].set_colors(np.array(['r','b'])[mask_clf])
violin_parts['cmaxes'].set_colors(np.array(['r','b'])[mask_clf])


violin_parts = ax[1].violinplot(results_reg.T)
ax[1].set_xticks(np.arange(1,13), complexity_funs_reg, rotation=45)
ax[1].set_ylabel('regression complexity')


for p_id, pc in enumerate(violin_parts['bodies']):
    pc.set_color(['r','b'][mask_reg[p_id]])
violin_parts['cbars'].set_colors(np.array(['r','b'])[mask_reg])
violin_parts['cmins'].set_colors(np.array(['r','b'])[mask_reg])
violin_parts['cmaxes'].set_colors(np.array(['r','b'])[mask_reg])



for aa in ax:
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    aa.grid(ls=':')
    aa.set_ylim(0,1)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/e0_common.png')
plt.savefig('figures/e0_common.pdf')
plt.savefig('figures/e0_common.eps')

