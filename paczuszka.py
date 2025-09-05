from problexity.classification import f1, n1, clsCoef, l2
import numpy as np
import matplotlib.pyplot as plt

datasets = np.load('Experiments/res/combined_datasets.npy')
print(datasets.shape) #(10, 5, 11, 350, 21)

datasets = datasets[:,:,-1]
print(datasets.shape) #(10, 5, 350, 21)

datasets = datasets.reshape(-1,350, 21)

s=20
random_idx = np.random.choice(50, size=s, replace=False)
datasets = datasets[random_idx]
print(datasets.shape) #(15, 350, 21)

measures = [f1, n1, clsCoef, l2]
complexities = []
for i in range(s):
    d = datasets[i]
    X = d[:,:20]
    y = d[:,-1]
    
    c = [m(X, y) for m in measures]
    complexities.append(c)

complexities = np.array(complexities)
print(complexities.shape) # s, 4

fig, ax = plt.subplots(4,4,figsize=(10,10))

for i in range(4):
    for j in range(4):
        
        ax[i,j].scatter(complexities[:,j], complexities[:,i], color='white', edgecolor='k', alpha=0.75)
        if i==3:
            ax[i,j].set_xlabel(measures[j].__name__)
        if j==0:
            ax[i,j].set_ylabel(measures[i].__name__)
        
plt.tight_layout()
plt.savefig('foo.png')

np.save('paczuszka.npy', datasets)
print(datasets.shape) #(15, 350, 21)


    
    
    
