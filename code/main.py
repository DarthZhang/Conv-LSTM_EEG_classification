import os
from src import data as dt
from src.classifier import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# Data importing...

subjects = [25,33]
path_to_data = '/home/moskaleona/alenadir/data/rawData' #'C:/Users/alena/Desktop/homed/laba/data/rawData' #'../sample_data'

data = dt.DataBuildClassifier(path_to_data).get_data(subjects, shuffle=True, random_state=1,
                                                     resample_to=128, windows=[(0.2, 0.5)],
                                                     baseline_window=(0.2, 0.3))
cv = StratifiedShuffleSplit(n_splits=4, test_size = 0.25, random_state = 108)

'''
param_grid = {
    'n_iter' : [100, 200, 300],
    'l1' : [0., 0.2, 0.4, 0.6],
    'l2' : [0., 0.2, 0.4, 0.6],
    'dropout' : [0., 0.2, 0.4, 0.6],
    'dropout_lstm' : [0., 0.2, 0.4, 0.6],
    'recurrent_dropout' : [0., 0.2, 0.4, 0.6],
}
'''
param_grid = {
    'n_iter' : [3, 2],
    'l1' : [0., 0.2]
}
if not os.path.isdir(os.path.join(os.pardir, 'results')):
    os.mkdir(os.path.join(os.pardir, 'results'))

for i in subjects:
    path_to_results = os.path.join(os.pardir, 'results', str(i))
    X, y = data[i][0], data[i][1]
    X_train, X_test, y_train, y_test = train_test_split(data[i][0], data[i][1], test_size=0.2, stratify=data[i][1], random_state=108)

    # Grid search
    clf = CnnLstmClassifier()
    gs = GridSearch(clf, param_grid, cv=cv, verbose=1)
    gs.fit(X_train, y_train, X_test, y_test, path_to_results=path_to_results)
