import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
from textwrap import wrap

df = pd.read_csv('datasets/Attractiveness_Analytics.csv', delimiter=',', quotechar='"')

if(not os.path.exists('plots')):
    dirname = 'plots'
    os.mkdir(dirname)
    dirname = 'Attractiveness_Analytics'
    os.mkdir('plots/%s' %dirname)

if(not os.path.exists('plots/Attractiveness_Analytics')):
    dirname = 'Attractiveness_Analytics'
    os.mkdir('plots/%s' %dirname)
    dirname = 'SVM'
    os.mkdir('plots/Attractiveness_Analytics/%s' %dirname)

if(not os.path.exists('plots/Attractiveness_Analytics/SVM')):
    dirname = 'SVM'
    os.mkdir('plots/Attractiveness_Analytics/%s' %dirname)

df = df.drop('image_id', 1)
X = df.ix[:, df.columns != 'Attractive']
y = df['Attractive']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30, random_state=30)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

param_grid = param_grid = {
    'C': [0.01, 0.1, 1, 2, 3,4,5,6,7,8,9,10],
    'kernel': ['linear', 'rbf', 'poly'],
}
clf = svm.SVC()
grid_search = GridSearchCV(clf, param_grid,
                       cv=5, n_jobs=-1)
results = grid_search.fit(X_train, y_train)

# make the predictions
y_pred = grid_search.predict(X_test)

print('Best params')
print(grid_search.best_params_)