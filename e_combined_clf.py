import numpy as np
from problexity.classification import f1, f3, f4, l2, n1, n3, n4, t1, clsCoef, hubs, t4
import matplotlib.pyplot as plt
from sklearn import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

np.random.seed(188)

combined_datasets = np.load('res/combined_datasets.npy')
        
complexity_funs = [f1, f3, f4, l2, n1, n3, n4, t1, clsCoef, hubs, t4]
n_targets = 5
n_datasets = 12

clfs = [
    KNeighborsClassifier(n_neighbors=5),
    DecisionTreeClassifier(random_state=2938),
    GaussianNB(),
    MLPClassifier(random_state=2322),
    SVC(random_state=3882)
]

print(combined_datasets.shape)
# (10, 5, 12, 500, 21)

res_clf = np.zeros((10, 12, 5, 10, len(clfs)))

for rep in range(10):
    for dataset_id in range(12):
        
        for target_com_id in range(5):
            
            X = combined_datasets[rep, target_com_id, dataset_id, :, :20]            
            y = combined_datasets[rep, target_com_id, dataset_id, :, -1]
            
            print(X.shape, y.shape)
            
            rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
            for fold, (train, test) in enumerate(rskf.split(X, y)):
                for clf_id, clf in enumerate(clfs):
                    clf_clone = clone(clf)
                    clf_clone.fit(X[train], y[train])
                    acc = accuracy_score(y[test], clf_clone.predict(X[test]))
                    
                    res_clf[rep, dataset_id, target_com_id, fold, clf_id] = acc
            
                print(res_clf[rep, dataset_id, target_com_id, fold])
                np.save('res/combined_clf.npy', res_clf)
                    