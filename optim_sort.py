import numpy as np
from sklearn.datasets import make_classification, make_blobs
from problexity.classification import f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4, t1, lsc, \
    density, clsCoef, hubs, t2, t3, t4, c1, c2
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

### genetic params
pop_size = 210
iters = 100
cross_ratio = 0.2
mut_ratio = 0.1
mut_std = 0.2

decay_cross = 0.01
decay_mut = 0.01

### target
target_X, target_y = load_breast_cancer(return_X_y=True)

complexity_fun = [f1, f1v, n3]
target_complexity = [f(target_X, target_y) for f in complexity_fun]
complexity_fun_names = [c.__name__ for c in complexity_fun]

### init
n_features = 10
n_features_target = 2
X, y = make_blobs(n_samples=500, n_features=n_features, centers=2)

population = np.random.normal(loc=0, scale=3, size=(pop_size, n_features, n_features_target))

pop_scores = np.full((pop_size, len(target_complexity)), np.nan)
pop_scores_all = []
fitness_all = []
order_all = []

### optimize
for i in range(iters): 
    # if i==3:
    #     exit()
    
    # score
    if i==0:
        for projection_id, projection in enumerate(population):
            pX = X@projection
            pX /= n_features
            
            for cf_id, cf in enumerate(complexity_fun):
                pop_scores[projection_id, cf_id] = np.abs(target_complexity[cf_id] - cf(pX, y))
        
        
    #reorder
    order = np.argsort(np.sum(pop_scores, axis=1))
    print(order)
    population = population[order]
    pop_scores = pop_scores[order]
    print('fitness;', np.sum(pop_scores, axis=1))
    
    
    pop_scores_all.append(np.copy(pop_scores))
    fitness_all.append(np.copy(np.sum(pop_scores, axis=1)))
    order_all.append(np.copy(order))


    if i!= iters-1: # w ostatnim kroku bez modyfikacji
                
        ### krzyżowanie
        n_crosses = np.max([int(cross_ratio*pop_size), 1])
        # n_dims = np.max([int(cross_factor*n_features_target), 1])
        
        new = []
        for n_cross_id in range(n_crosses):
            # dim_to_cross = np.random.choice(n_features_target, n_dims)
            
            selected = np.copy(population[n_cross_id])
            another = np.copy(population[np.random.choice(pop_size)])
            w = np.random.normal(0.7, 0.2)
            crossed = w*selected + (1-w)*another

            new.append(crossed)
            
            # score new
            score_new = [[] for i in range(len(target_complexity))]
            for projection_id, projection in enumerate(new):
                pX = X@projection
                pX /= n_features
                
                for cf_id, cf in enumerate(complexity_fun):
                    score_new[cf_id].append(np.abs(target_complexity[cf_id] - cf(pX, y)))
            
            
        population[-n_crosses:] = new
        pop_scores[-n_crosses:] = np.array(score_new).swapaxes(0,1)
        cross_ratio = cross_ratio*(1-decay_cross)
            
        ### mutacja
        n_mutations = int(mut_ratio*pop_size)
        arg_to_mut = np.random.choice(np.arange(pop_size), n_mutations)
        for arg in arg_to_mut:
            population[arg] += np.random.normal(loc=0, scale=mut_std)
            # score mutated
            pX = X@projection
            pX /= n_features
            
            for cf_id, cf in enumerate(complexity_fun):
                pop_scores[arg, cf_id] = (np.abs(target_complexity[cf_id] - cf(pX, y)))
            
        mut_ratio = mut_ratio*(1-decay_mut)
        
        uniq, c = np.unique(population, return_counts=True, axis=0)
        print(c)

        fitness = np.sum(pop_scores, axis=1)
        
        



# get best
proj_best = population[0]
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
ax[0].set_title('measures: %s \n target: %s result: %s \n score: %0.3f' % 
                (complexity_fun_names, target_complexity, np.round(projection_complexity,3), fitness[0]))

cols = plt.cm.coolwarm(np.linspace(0,1,best_over_time.shape[1]))
for i in range(best_over_time.shape[1]):
    ax[1].scatter(np.arange(iters), best_over_time[:,i], c=cols[i], s=5)
    ax[1].plot(np.arange(iters), mean_over_time[:,i], c=cols[i], label=complexity_fun_names[i])
    ax[1].fill_between(np.arange(iters), 
                    mean_over_time[:,i]-div_over_time[:,i],
                    mean_over_time[:,i]+div_over_time[:,i],
                    color=cols[i], lw=0, alpha=0.1)

ax[2].imshow(np.array(fitness_all).T, cmap='coolwarm', aspect='auto')


# diverse = np.argsort(-np.std(pop_scores_all, axis=(0,1)))[:3]
# print(diverse)
# img = np.array(pop_scores_all)[:,:,diverse].swapaxes(0,1)
# img -= np.min(img)
# img /= np.max(img) + 0.0001
# ax[3].imshow(img, aspect='auto')

ax[3].imshow(np.array(order_all).T, aspect='auto', cmap='coolwarm',)

ax[1].legend()

for aa in ax:
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    aa.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')