from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt


x0,y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, n_repeated=0)

proj = np.random.normal(loc=0, scale=3.0, size=(2,2))

x1 = x0@proj.T

fig, ax = plt.subplots(1,2, figsize=(10,5))

ax[0].scatter(x0[:,0], x0[:,1], c=y, cmap='coolwarm')
ax[1].scatter(x1[:,0], x1[:,1], c=y, cmap='coolwarm')

for aa in ax:
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    aa.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')