import numpy as np
import matplotlib.pyplot as plt

clfs = [ 'KNN', 'DT', 'GNB', 'MLP', 'SVM']
res_clf = np.load('res/combined_clf.npy')

print(res_clf.shape) # (10, 12, 5, 10, 5)

res_clf = res_clf.reshape(-1, 5, 10, 5)
res_clf = np.mean(res_clf, axis=2)

print(res_clf.shape) # (reps*datasets, targets, clfs)

fig, ax = plt.subplots(1, 5, figsize=(10,4), sharex=True, sharey=True)

for clf_id, clf in enumerate(clfs):
    
    rc = res_clf[:,:,clf_id]
    mrc = np.mean(rc, axis=0)
    mrc -= 0.5
    mrc /=0.5
    colors = plt.cm.coolwarm(mrc)

    ax[clf_id].set_title('%s' % (clf))
    bplot = ax[clf_id].boxplot(rc, patch_artist=True)
    
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        
    for median, color in zip(bplot['medians'], colors):
        median.set_color('black')
        median.set_ls(':')
    
    ax[clf_id].set_xticks(np.arange(5)+1, ['easy', 'med-easy', 'medium', 'med-complex', 'complex'], rotation=90)

    
for aa in ax:
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    aa.grid(ls=':')
    
ax[0].set_ylabel('accuracy score')


plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/clf-scores.png')
plt.savefig('figures/clf-scores.pdf')
plt.savefig('figures/clf-scores.eps')
