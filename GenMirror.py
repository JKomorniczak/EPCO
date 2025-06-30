import numpy as np
from sklearn.datasets import make_classification, make_blobs
from problexity.classification import f1, f1v, f2, f3, f4, l1, l2, l3, n1, n2, n3, n4, t1, lsc, \
    density, clsCoef, hubs, t2, t3, t4, c1, c2
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from tqdm import tqdm

class GenMirror:
    def __init__(self, X_source, y_source, X_target, y_target, measures):
        self.X_source = X_source
        self.y_source = y_source
        self.X_target = X_target
        self.y_target = y_target

        self.measures = measures
        
        self.target_complexity = [f(self.X_target, self.y_target) for f in self.measures]

        
    def generate(self, pop_size = 100, iters=100, cross_ratio=0.3, mut_ratio=0.1, mut_std = 0.1, decay = 0.01):
        self.iters = iters
        self.order_all = []
        self.scores_all = []
        self.measures_all = []
                
        self.population = np.random.normal(loc=0, scale=3, size=(pop_size, self.X_source.shape[1], self.X_source.shape[1]))
        self.pop_scores = np.full((pop_size, len(self.measures)), np.nan)
        
        ### optimize
        for i in tqdm(range(self.iters)):
            
            # score
            if i==0:
                for projection_id, projection in enumerate(self.population):
                    pX = self.project(projection)
                    
                    for cf_id, cf in enumerate(self.measures):
                        self.pop_scores[projection_id, cf_id] = np.abs(self.target_complexity[cf_id] - cf(pX, self.y_source))
                                        
            
            order = np.zeros((pop_size)).astype(int)
            indexes = np.arange(0, pop_size-len(self.measures)+1, len(self.measures)+1) # additional for mean criteria
            
            
            for m in range(len(self.measures)):
                r = np.argsort(self.pop_scores[:,m])
                order[indexes+m] = r[:len(indexes)]
            
            r = np.argsort(np.sum(self.pop_scores, axis=1))
            indexes2 = indexes+len(self.measures)
            indexes2 = indexes2[indexes2<pop_size]
            order[indexes2] = r[:len(indexes2)]            
            
            print(np.unique(order, return_counts=True))
            # exit()
            self.population = self.population[order]
            self.pop_scores = self.pop_scores[order]
            
            ## for vis
            self.order_all.append(order)
            self.scores_all.append(np.sum(self.pop_scores, axis=1))
            self.measures_all.append(self.pop_scores)
            
            if i!= self.iters-1: # w ostatnim kroku bez modyfikacji
                        
                ### krzyżowanie
                n_crosses = np.max([int(cross_ratio*pop_size), 1])                
                new = []
                for n_cross_id in range(n_crosses):                   
                    selected = np.copy(self.population[n_cross_id])
                    another = np.copy(self.population[np.random.choice(pop_size)])
                    
                    w = np.random.normal(0.7, 0.2)
                    crossed = w*selected + (1-w)*another
                    new.append(crossed)
                    
                    # score new
                    score_new = [[] for i in range(len(self.target_complexity))]
                    for projection_id, projection in enumerate(new):
                        pX = self.project(projection)               
                        for cf_id, cf in enumerate(self.measures):
                            score_new[cf_id].append(np.abs(self.target_complexity[cf_id] - cf(pX, self.y_source)))
                    
                    
                self.population[-n_crosses:] = new
                self.pop_scores[-n_crosses:] = np.array(score_new).swapaxes(0,1)
                cross_ratio = cross_ratio*(1-decay)
                    
                ### mutacja
                n_mutations = int(mut_ratio*pop_size)
                arg_to_mut = np.random.choice(np.arange(pop_size), n_mutations)
                
                for arg in arg_to_mut:
                    self.population[arg] += np.random.normal(loc=0, scale=mut_std)
                    
                    # score mutated
                    pX = self.project(self.population[arg])
                
                    for cf_id, cf in enumerate(self.measures):
                        self.pop_scores[arg, cf_id] = (np.abs(self.target_complexity[cf_id] - cf(pX, self.y_source)))
                    
                mut_ratio = mut_ratio*(1-decay)

    def return_best(self, index=0):
        return self.project(self.population[index]), self.y_source

    def project(self, projection):
        pX = self.X_source@projection
        pX = pX/self.X_source.shape[1]
        return pX
    
    def gen_image(self):
        bestX, y = self.return_best()
        complexity = [cf(X, y) for cf in self.measures]
        complexity_fun_names = [c.__name__ for c in self.measures]
        best_over_time = np.min(np.array(self.measures_all), axis=1)
        mean_over_time = np.mean(np.array(self.measures_all), axis=1)
        div_over_time = np.std(np.array(self.measures_all), axis=1)
        
        print(best_over_time.shape)
        
        fig, ax = plt.subplots(4,1,figsize=(10,10))

        ax[0].scatter(bestX[:,0], bestX[:,1], c=y, cmap='coolwarm')
        ax[0].set_title('measures: %s \n target: %s result: %s \n score: %0.3f' % 
                        (complexity_fun_names, np.round(self.target_complexity,3), np.round(complexity,3), np.sum(self.pop_scores[0])))

        cols = plt.cm.coolwarm(np.linspace(0,1,len(self.measures)))
        for i in range(len(self.measures)):
            ax[1].scatter(np.arange(self.iters), best_over_time[:,i], color=cols[i], s=5)
            ax[1].plot(np.arange(self.iters), mean_over_time[:,i], c=cols[i], label=complexity_fun_names[i])
            ax[1].fill_between(np.arange(self.iters), 
                            mean_over_time[:,i]-div_over_time[:,i],
                            mean_over_time[:,i]+div_over_time[:,i],
                            color=cols[i], lw=0, alpha=0.1)

        ax[2].imshow(np.array(self.scores_all).T, cmap='coolwarm', aspect='auto')
        ax[2].set_ylabel('mean fitness')
        ax[3].imshow(np.array(self.order_all).T, aspect='auto', cmap='coolwarm', interpolation='none')
        ax[3].set_ylabel('order')

        ax[1].legend()

        for aa in ax:
            aa.spines['top'].set_visible(False)
            aa.spines['right'].set_visible(False)
            aa.grid(ls=':')

        plt.tight_layout()
        plt.savefig('foo.png')
        
    def gen_pareto(self):
        aa = np.array(self.measures_all)
        print(aa.shape) #(600, 50, 2)
                
        fig, ax = plt.subplots(1,1,figsize=(10,10))

        cols = plt.cm.coolwarm(np.linspace(0,1,self.iters))
        for iter in range(self.iters):
            ax.scatter(aa[iter,:,0], aa[iter,:,1], color=cols[iter], alpha=0.2)
        ax.scatter(0,0,c='k',marker='x')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(ls=':')

        plt.tight_layout()
        plt.savefig('foo2.png')



### target
X_target, y_target = load_breast_cancer(return_X_y=True)
# X_source, y_source = make_classification(n_samples=200, n_features=2, n_informative=2, n_repeated=0, n_redundant=0)
X_source, y_source = make_classification(n_samples=200)
complexity_fun = [f1, n3]

# optimize
mirror = GenMirror(X_source, y_source, X_target, y_target, complexity_fun)
mirror.generate(iters=100, pop_size=70, cross_ratio=0.25, mut_ratio=0.1)

X, y = mirror.return_best()

mirror.gen_image()
mirror.gen_pareto()

