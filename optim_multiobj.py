import numpy as np
from sklearn.datasets import make_classification
from problexity.classification import f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4, t1, lsc, density, clsCoef, hubs, t2, t3, t4, c1, c2
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

### genetic params
pop_size = 150
iters = 50
cross_ratio = 0.8
mut_ratio = 0.3
mut_std = 0.1

decay_cross = 0.1
decay_mut = 0.1

### target
target_X, target_y = load_breast_cancer(return_X_y=True)

complexity_fun = [f1v, n3, t4, l2]
target_complexity = [f(target_X, target_y) for f in complexity_fun]
complexity_fun_names = [c.__name__ for c in complexity_fun]

### init
n_features = 10
n_features_target = 2
X, y = make_classification(n_samples=500,
                           n_features=n_features, 
                           n_informative=n_features//2, 
                           n_redundant=n_features//2, n_repeated=0, 
                           n_clusters_per_class=1)

population = np.random.normal(loc=0, scale=1, size=(pop_size, n_features, n_features_target))

pop_scores = np.full((pop_size, len(target_complexity)), np.nan)
pop_scores_all = []
fitness_all = []

### optimize
for i in range(iters):    
    for projection_id, projection in enumerate(population):
        pX = X@projection
        pX /= n_features
        
        for cf_id, cf in enumerate(complexity_fun):
            pop_scores[projection_id, cf_id] = np.abs(target_complexity[cf_id] - cf(pX, y))
    
    #TODO
    fitness = np.sum(pop_scores, axis=1)
    print(fitness)
    
    pop_scores_all.append(np.copy(pop_scores))
    fitness_all.append(np.copy(fitness))

    if i!= iters-1: # w ostatnim kroku bez modyfikacji
                
        ### krzyżowanie
        n_crosses = np.max([int(cross_ratio*pop_size), 1])
        arg_sorted = np.argsort(fitness)
        arg_to_cross = arg_sorted[:n_crosses]
        arg_to_replace = np.random.choice(arg_sorted[n_crosses:], n_crosses)
        arg_pairs_cross = np.array([np.random.permutation(arg_to_cross), arg_to_cross]).swapaxes(0,1)

        for n_cross_id, (arg1, arg2) in enumerate(arg_pairs_cross):
            crossed = (population[arg1] + population[arg2])/2
            #zastępujemy
            population[arg_to_replace[n_cross_id]] = crossed
            
        cross_ratio = cross_ratio*(1-decay_cross)
        print(cross_ratio, n_crosses)
            
        ### mutacja
        n_mutations = int(mut_ratio*pop_size)
        arg_to_mut = np.random.choice(np.arange(pop_size), n_mutations)
        for arg in arg_to_mut:
            population[arg] += np.random.normal(loc=0, scale=mut_std)
            
        mut_ratio = mut_ratio*(1-decay_mut)
        print(mut_ratio, n_mutations)


# get best
arg_best = np.argmin(fitness)
proj_best = population[arg_best]
bestX = X@proj_best
bestX /= n_features
projection_complexity = np.array([cf(bestX, y) for cf in complexity_fun])

# visualize
pop_scores_all = np.array(pop_scores_all) 
print(pop_scores_all.shape)  # 200, 50, 3

best_over_time = np.min(pop_scores_all, axis=1) # 200, 3

mean_over_time = np.mean(pop_scores_all, axis=1) # 200, 3
div_over_time = np.std(pop_scores_all, axis=1) # 200, 3

print(best_over_time.shape, np.arange(iters).shape)

###

fig, ax = plt.subplots(4,1,figsize=(10,10))

ax[0].scatter(bestX[:,0], bestX[:,1], c=y, cmap='coolwarm')
print(complexity_fun_names, target_complexity, np.round(projection_complexity,3), fitness, arg_best)
ax[0].set_title('measures: %s \n target: %s result: %s \n score: %0.3f' % 
                (complexity_fun_names, target_complexity, np.round(projection_complexity,3), fitness[arg_best]))

cols = plt.cm.coolwarm(np.linspace(0,1,best_over_time.shape[1]))
for i in range(best_over_time.shape[1]):
    ax[1].scatter(np.arange(iters), best_over_time[:,i], c=cols[i], s=5)
    ax[1].plot(np.arange(iters), mean_over_time[:,i], c=cols[i], label=complexity_fun_names[i])
    ax[1].fill_between(np.arange(iters), 
                    mean_over_time[:,i]-div_over_time[:,i],
                    mean_over_time[:,i]+div_over_time[:,i],
                    color=cols[i], lw=0, alpha=0.1)
    
ax[2].imshow(np.array(fitness_all).T, cmap='coolwarm', aspect='auto')
ax[3].imshow(np.array(pop_scores_all)[:,:,:3].swapaxes(0,1), aspect='auto')

ax[1].legend()

for aa in ax:
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    aa.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')