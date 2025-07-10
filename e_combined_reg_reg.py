import numpy as np
from sklearn import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

np.random.seed(188)

combined_datasets = np.load('res/combined_datasets_reg.npy')

regs = [
    KNeighborsRegressor(n_neighbors=5),
    DecisionTreeRegressor(random_state=2938),
    LinearRegression(random_state=2322),
    MLPRegressor(random_state=3882),
    SVR(random_state=3882)
]

print(combined_datasets.shape)
# (10, 5, 12, 500, 21)
exit()

res_reg = np.zeros((10, 6, 5, 10, len(regs)))

for rep in range(10): 
    for dataset_id in range(12):
        
        for target_com_id in range(5):
            
            X = combined_datasets[rep, target_com_id, dataset_id, :, :20]            
            y = combined_datasets[rep, target_com_id, dataset_id, :, -1]
            
            print(X.shape, y.shape)
            
            rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
            for fold, (train, test) in enumerate(rskf.split(X, y)):
                for reg_id, reg in enumerate(regs):
                    reg_clone = clone(reg)
                    reg_clone.fit(X[train], y[train])
                    acc = mean_absolute_error(y[test], reg_clone.predict(X[test]))
                    
                    res_reg[rep, dataset_id, target_com_id, fold, reg_id] = acc
            
                print(res_reg[rep, dataset_id, target_com_id, fold])
                np.save('res/combined_reg_reg.npy', res_reg)
                    