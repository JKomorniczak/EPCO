import numpy as np
from problexity.classification import f1, f3, f4, l2, n1, n3, n4, t1, clsCoef, hubs, t4
import matplotlib.pyplot as plt

clfs = [ 'KNN', 'DT', 'GNB', 'MLP', 'SVM']
res_clf = np.load('res/combined_clf.npy')

print(res_clf.shape) # (10, 12, 5, 10, 5)

res_clf = res_clf.reshape(-1, 5, 10, 5)
res_clf = np.mean(res_clf, axis=2)

print(res_clf.shape) # (reps*datasets, targets, clfs)

fig, ax = plt.subplots(5, 1, figsize=(10,10))

for clf_id, clf in enumerate(clfs):
    ax[clf_id].set_title(clf)
    ax[clf_id].violinplot(res_clf[:,:,clf_id])
    
plt.tight_layout()
plt.savefig('foo.png')
