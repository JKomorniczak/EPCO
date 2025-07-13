import numpy as np
import matplotlib.pyplot as plt


regs = [ 'KNR', 'DTR', 'BRR', 'MLPR', 'SVMR']
res_reg = np.load('res/e2_combined_reg.npy')

measures = ['MAE', 'MSE', 'R2']
measure_id = 0

res_reg = res_reg[...,measure_id]
print(res_reg.shape) # (10, 5, 5, 10, 5)

res_reg = res_reg.reshape(-1, 5, 10, 5)
res_reg = np.mean(res_reg, axis=2)

print(res_reg.shape) # (reps*datasets, targets, clfs)

fig, ax = plt.subplots(1, 5, figsize=(10,4), sharex=True, sharey=True)

for reg_id, reg in enumerate(regs):
    
    rc = res_reg[:,:,reg_id]
    mrc = np.mean(rc, axis=0)
    mrc -= np.min(mrc)
    mrc /= np.max(mrc)
    colors = plt.cm.coolwarm(mrc)

    ax[reg_id].set_title('%s' % (reg))
    bplot = ax[reg_id].boxplot(rc, patch_artist=True)
    
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        
    for median, color in zip(bplot['medians'], colors):
        median.set_color('black')
        median.set_ls(':')
    
    ax[reg_id].set_xticks(np.arange(5)+1, ['easy', 'med-easy', 'medium', 'med-complex', 'complex'], rotation=90)

    
for aa in ax:
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    aa.grid(ls=':')
    
ax[0].set_ylabel(measures[measure_id])


plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/reg-scores.png')
plt.savefig('figures/reg-scores.pdf')
plt.savefig('figures/reg-scores.eps')
