import numpy as np
from sklearn import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

np.random.seed(188)

combined_datasets = np.load('vapor/res_prev/combined_datasets_reg_f.npy')

regs = [
    KNeighborsRegressor(n_neighbors=5),
    DecisionTreeRegressor(random_state=2938),
    LinearRegression(),
    MLPRegressor(random_state=3882),
    SVR()
]

print(combined_datasets.shape)
# (10, 5, 5, 350, 21)

res_reg = np.zeros((10, 5, 5, 10, len(regs), 3))

for rep in range(10):
    for dataset_id in range(5):
        
        for target_com_id in range(5):
            
            X = combined_datasets[rep, target_com_id, dataset_id, :, :20]            
            y = combined_datasets[rep, target_com_id, dataset_id, :, -1]
            
            print(X.shape, y.shape)
            
            rskf = RepeatedKFold(n_splits=2, n_repeats=5)
            for fold, (train, test) in enumerate(rskf.split(X, y)):
                for reg_id, reg in enumerate(regs):
                    reg_clone = clone(reg)
                    reg_clone.fit(X[train], y[train])
                    for m_id, m in enumerate([mean_absolute_error, mean_squared_error, r2_score]):
                        acc = m(y[test], reg_clone.predict(X[test]))
                        res_reg[rep, dataset_id, target_com_id, fold, reg_id, m_id] = acc
            
                print(res_reg[rep, dataset_id, target_com_id, fold])
                np.save('res/combined_reg_reg_f.npy', res_reg)
                    