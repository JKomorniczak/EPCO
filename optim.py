import numpy as np
from sklearn.datasets import make_classification
from problexity.classification import f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4, t1, lsc, density, clsCoef, hubs, t2, t3, t4, c1, c2
import matplotlib.pyplot as plt

n_features = 2
n_features_target = 2
X, y = make_classification(n_samples=500, n_features=n_features, n_informative=n_features, n_redundant=0, n_repeated=0)

### genetic params
pop_size = 150
iters = 100
cross_ratio = 0.2
mut_ratio = 0.03
mut_std = 0.1

### target
complexity_fun = [f1, f4, l1, l2, n3]
target_complexity = np.array([0.576, 0.1, 0.3, 0.3, 0.2])
complexity_fun_names = [c.__name__ for c in complexity_fun]


### init
population = np.random.normal(loc=0, scale=1, size=(pop_size, n_features, n_features_target))
pop_scores = np.full((pop_size), np.nan)

pop_scores_all = []

### optimize
for i in range(iters):    
    for projection_id, projection in enumerate(population):
        pX = X@projection
        pX /= n_features
        
        projection_complexity = np.array([cf(pX, y) for cf in complexity_fun])
        pop_scores[projection_id] = np.sum(np.abs(target_complexity - projection_complexity)) 
        # todo sprytniej ogarnąć wielokryterialne niż suma
    
    # dla wizualizacji
    pop_scores_all.append(np.copy(pop_scores))
    print(np.min(pop_scores))
        
    if i!= iters-1: # w ostatnim kroku bez modyfikacji
        
        ### krzyżowanie
        n_crosses = int(cross_ratio*pop_size)
        arg_sorted = np.argsort(pop_scores)
        arg_to_cross = arg_sorted[:n_crosses]
        arg_to_replace = np.random.choice(arg_sorted[n_crosses:], n_crosses)
        arg_pairs_cross = np.array([np.random.permutation(arg_to_cross), arg_to_cross]).swapaxes(0,1)

        for n_cross_id, (arg1, arg2) in enumerate(arg_pairs_cross):
            crossed = (population[arg1] + population[arg2])/2
            #zastępujemy
            population[arg_to_replace[n_cross_id]] = crossed
            
        ### mutacja
        n_mutations = int(cross_ratio*pop_size)
        arg_to_mut = np.random.choice(np.arange(pop_size), n_mutations)
        for arg in arg_to_mut:
            population[arg] += np.random.normal(loc=0, scale=mut_std)


# get best
arg_best = np.argmin(pop_scores)
proj_best = population[arg_best]
bestX = X@proj_best
bestX /= n_features
projection_complexity = np.array([cf(bestX, y) for cf in complexity_fun])

# visualize
pop_scores_all = np.array(pop_scores_all)
best_over_time = np.min(pop_scores_all, axis=1)

mean_over_time = np.mean(pop_scores_all, axis=1)
div_over_time = np.std(pop_scores_all, axis=1)

print(best_over_time.shape, np.arange(iters).shape)
###

fig, ax = plt.subplots(2,1,figsize=(10,10))

ax[0].scatter(bestX[:,0], bestX[:,1], c=y)
ax[0].set_title('measures: %s \n target: %s result: %s \n score: %0.3f' % 
                (complexity_fun_names, target_complexity, np.round(projection_complexity,3), pop_scores[arg_best]))

ax[1].scatter(np.arange(iters), best_over_time, c='r')
ax[1].plot(np.arange(iters), mean_over_time, c='b')
ax[1].fill_between(np.arange(iters), 
                   mean_over_time-div_over_time, 
                   mean_over_time+div_over_time,
                   color='b', lw=0, alpha=0.1)

for aa in ax:
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    aa.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')