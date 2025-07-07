import numpy as np
import matplotlib.pyplot as plt

clfs = [ 'KNN', 'DT', 'GNB', 'MLP', 'SVM']
res_clf = np.load('res/combined_clf.npy')

print(res_clf.shape) # (10, 12, 5, 10, 5)

res_clf = res_clf.reshape(-1, 5, 10, 5)
res_clf = np.mean(res_clf, axis=2)

print(res_clf.shape) # (reps*datasets, targets, clfs)

fig, ax = plt.subplots(5, 1, figsize=(10,10), sharex=True, sharey=True)

for clf_id, clf in enumerate(clfs):
    
    rc = res_clf[:,:,clf_id]
    mrc = np.mean(rc, axis=0)
    mrc -= 0.5
    mrc /=0.5
    colors = plt.cm.coolwarm(mrc)

    ax[clf_id].set_ylabel('%s \naccuracy score' % (clf))
    bplot = ax[clf_id].boxplot(rc, patch_artist=True)
    
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
for aa in ax:
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    aa.grid(ls=':')
    
ax[-1].set_xticks(np.arange(5)+1, ['easy', 'mid/easy', 'medium', 'mid/complex', 'complex'])
    
plt.tight_layout()
plt.savefig('foo.png')
