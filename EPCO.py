import numpy as np
from tqdm import tqdm

class EPCO:
    def __init__(self, X_source, y_source, target_complexity, measures, vis=False):
        self.X_source = X_source
        self.y_source = y_source
        self.measures = measures
        self.target_complexity = target_complexity
        self.vis = vis
        
        if len(target_complexity) != len(measures):
            raise AttributeError('Wrong length of target complexity in relation to measures')

    def generate(self, pop_size = 100, iters=100, cross_ratio=0.25, mut_ratio=0.1, mut_std = 0.1, decay = 0.007):
        self.iters = iters
                
        self.population = np.random.normal(loc=0, scale=3, size=(pop_size, self.X_source.shape[1], self.X_source.shape[1]))
        self.pop_scores = np.full((pop_size, len(self.measures)), np.nan)
        
        if self.vis:
            self.order_all = []
            self.scores_all = []
            self.measures_all = []
            
        ### optimize
        for i in tqdm(range(self.iters)):
            
            # score
            if i==0:
                for projection_id, projection in enumerate(self.population):
                    pX = self.project(projection)
                    
                    for cf_id, cf in enumerate(self.measures):
                        self.pop_scores[projection_id, cf_id] = np.abs(self.target_complexity[cf_id] - cf(pX, self.y_source))
                                        
            
            order = np.zeros((pop_size)).astype(int)
            if len(self.measures)==1:
                n = len(self.measures)
            else:
                n = len(self.measures)+1 # additional for sum criteria
            indexes = np.arange(0, pop_size-n, n) 
            
            for m in range(len(self.measures)):
                r = np.argsort(self.pop_scores[:,m])
                order[indexes+m] = r[:len(indexes)]
            
            if len(self.measures)>1:
                r = np.argsort(np.sum(self.pop_scores, axis=1))
                indexes2 = indexes+len(self.measures)
                indexes2 = indexes2[indexes2<pop_size]
                order[indexes2] = r[:len(indexes2)]         

            self.population = self.population[order]
            self.pop_scores = self.pop_scores[order]
            
            ## save for visualizaions -- optional
            if self.vis:
                self.order_all.append(order)
                self.scores_all.append(np.sum(self.pop_scores, axis=1))
                self.measures_all.append(self.pop_scores)
                
            if i!= self.iters-1: # last iteration not modifying the population
                        
                ### crossover
                n_crosses = np.max([int(cross_ratio*pop_size), 1])                
                new = []
                for n_cross_id in range(n_crosses):                   
                    selected = np.copy(self.population[n_cross_id])
                    another = np.copy(self.population[np.random.choice(pop_size)])
                    
                    w = np.random.normal(0.7, 0.2)
                    crossed = w*selected + (1-w)*another
                    new.append(crossed)
                    
                    # score new
                    score_new = [[] for i in range(len(self.measures))]
                    for projection_id, projection in enumerate(new):
                        pX = self.project(projection)               
                        for cf_id, cf in enumerate(self.measures):
                            score_new[cf_id].append(np.abs(self.target_complexity[cf_id] - cf(pX, self.y_source)))
                    
                self.population[-n_crosses:] = new
                self.pop_scores[-n_crosses:] = np.array(score_new).swapaxes(0,1)
                cross_ratio = cross_ratio*(1-decay)
                    
                ### mutation
                n_mutations = int(mut_ratio*pop_size)
                arg_to_mut = np.random.choice(np.arange(pop_size), n_mutations)
                
                for arg in arg_to_mut:
                    self.population[arg] += np.random.normal(loc=0, scale=mut_std, size=self.population[arg].shape)
                    
                    # score mutated
                    pX = self.project(self.population[arg])
                
                    for cf_id, cf in enumerate(self.measures):
                        self.pop_scores[arg, cf_id] = (np.abs(self.target_complexity[cf_id] - cf(pX, self.y_source)))
                    
                mut_ratio = mut_ratio*(1-decay)

    def return_best(self, index=None):
        if index==None:
            index = len(self.measures)
        return self.project(self.population[index]), self.y_source

    def project(self, projection):
        pX = self.X_source@projection
        pX = pX/self.X_source.shape[1]
        return pX