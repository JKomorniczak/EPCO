import numpy as np
from sklearn import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression

regs = [
    KNeighborsRegressor(n_neighbors=5),
    DecisionTreeRegressor(random_state=2938),
    LinearRegression(),
    MLPRegressor(random_state=3882),
    SVR()
]

X, y = make_regression(n_samples=500)

print(cross_val_score(regs[0], X, y, cv=10, scoring='neg_mean_absolute_error'))
exit()

rskf = RepeatedKFold(n_splits=2, n_repeats=5)
for fold, (train, test) in enumerate(rskf.split(X, y)):
    for reg_id, reg in enumerate(regs):
        reg_clone = clone(reg)
        reg_clone.fit(X[train], y[train])
        acc = mean_squared_error(y[test], reg_clone.predict(X[test]))
        
        print(acc)
        
        exit()
