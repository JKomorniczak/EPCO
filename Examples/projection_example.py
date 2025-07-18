from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from problexity.classification import f1, n1

np.random.seed(18881)

x0,y = make_classification(n_samples=500, n_features=10)

proj1 = np.random.normal(loc=0, scale=3.0, size=(10,10))
proj2 = np.random.normal(loc=0, scale=3.0, size=(10,10))

x1 = x0@proj1.T
x1 /= 10
x2 = x0@proj2.T
x2 /= 10

fig, ax = plt.subplots(1,3, figsize=(10,3.7), sharex=True, sharey=True)

x0pca = PCA(n_components=2).fit_transform(x0)
x1pca = PCA(n_components=2).fit_transform(x1)
x2pca = PCA(n_components=2).fit_transform(x2)

m0 = [m(x0, y) for m in [f1, n1]]
m1 = [m(x1, y) for m in [f1, n1]]
m2 = [m(x2, y) for m in [f1, n1]]

ax[0].scatter(x0pca[:,0], x0pca[:,1], c=y, cmap='coolwarm', s=10)
ax[1].scatter(x1pca[:,0], x1pca[:,1], c=y, cmap='coolwarm', s=10)
ax[2].scatter(x2pca[:,0], x2pca[:,1], c=y, cmap='coolwarm', s=10)

ax[0].set_title('original dataset \nF1 = %.2f | N1 = %.2f' % (m0[0], m0[1]))
ax[1].set_title('transformed with $p_0$ \nF1 = %.2f | N1 = %.2f' % (m1[0], m1[1]))
ax[2].set_title('transformed with $p_1$ \nF1 = %.2f | N1 = %.2f' % (m2[0], m2[1]))

for aa in ax:
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    aa.grid(ls=':')
    aa.set_xlabel('principal component 0')
    

ax[0].set_ylabel('principal component 1')
    
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/example_proj.png')
plt.savefig('figures/example_proj.eps')
plt.savefig('figures/example_proj.pdf')