import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import os

df = pd.read_csv('datasets/Attractiveness_Analytics.csv', delimiter=',', quotechar='"')

if(not os.path.exists('plots')):
    dirname = 'plots'
    os.mkdir(dirname)
    dirname = 'Attractiveness_Analytics'
    os.mkdir('plots/%s' %dirname)

if(not os.path.exists('plots/Attractiveness_Analytics')):
    dirname = 'Attractiveness_Analytics'
    os.mkdir('plots/%s' %dirname)
    dirname = 'DecisionTree'
    os.mkdir('plots/Attractiveness_Analytics/%s' %dirname)

if(not os.path.exists('plots/Attractiveness_Analytics/DecisionTree')):
    dirname = 'DecisionTree'
    os.mkdir('plots/Attractiveness_Analytics/%s' %dirname)

# print(df.head(5))

df = df.drop('image_id', 1)
X = df.ix[:, df.columns != 'Attractive']
y = df['Attractive']


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

param_grid = param_grid = {
    # 'min_samples_split': [3, 5, 10],
    # 'n_estimators' : [100, 300],
    'max_depth': [1,2,3,4,5,6,7,8,9,10,11,12,13,14, 15, 20],
    # 'max_features': [3, 5, 10, 20]
}
clf = DecisionTreeClassifier(random_state=7)
grid_search = GridSearchCV(clf, param_grid,
                       cv=5, n_jobs=-1)
results = grid_search.fit(X_train, y_train)

# make the predictions
y_pred = grid_search.predict(X_test)

print('Best params')
print(grid_search.best_params_)