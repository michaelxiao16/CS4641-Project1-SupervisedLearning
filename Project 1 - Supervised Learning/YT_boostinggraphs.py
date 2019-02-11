import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
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
    dirname = 'Boosting'
    os.mkdir('plots/Attractiveness_Analytics/%s' %dirname)

if(not os.path.exists('plots/Attractiveness_Analytics/Boosting')):
    dirname = 'Boosting'
    os.mkdir('plots/Attractiveness_Analytics/%s' %dirname)


# print(df.head(5))
df = df.drop('image_id', 1)
X = df.ix[:, df.columns != 'Attractive']
y = df['Attractive']


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30, random_state=30)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Boosting_ADABoost
train_size = len(X_train)
max_n_estimators = range(2, 31, 1)
train_err6 = [0] * len(max_n_estimators)
test_err6 = [0] * len(max_n_estimators)
train_err8 = [0] * len(max_n_estimators)
test_err8 = [0] * len(max_n_estimators)
train_err10 = [0] * len(max_n_estimators)
test_err10 = [0] * len(max_n_estimators)

# max_depths = [4, 6, 8]

for i, o in enumerate(max_n_estimators):
    print 'AdaBoostClassifier: learning a decision tree with n_estimators=' + str(o)
    dt4 = DecisionTreeClassifier(max_depth=6)
    dt6 = DecisionTreeClassifier(max_depth=8)
    dt8 = DecisionTreeClassifier(max_depth=10)
    bdt4 = AdaBoostClassifier(base_estimator=dt4, n_estimators=o)
    bdt6 = AdaBoostClassifier(base_estimator=dt6, n_estimators=o)
    bdt8 = AdaBoostClassifier(base_estimator=dt8, n_estimators=o)
    bdt4.fit(X_train, y_train)
    bdt6.fit(X_train, y_train)
    bdt8.fit(X_train, y_train)
    train_err6[i] = mean_squared_error(y_train,
                                     bdt4.predict(X_train))
    test_err6[i] = mean_squared_error(y_test,
                                    bdt4.predict(X_test))
    train_err8[i] = mean_squared_error(y_train,
                                     bdt6.predict(X_train))
    test_err8[i] = mean_squared_error(y_test,
                                    bdt6.predict(X_test))
    train_err10[i] = mean_squared_error(y_train,
                                     bdt8.predict(X_train))
    test_err10[i] = mean_squared_error(y_test,
                                    bdt8.predict(X_test))
    print '---'

# Plot results
print 'plotting results'
plt.figure()
title = 'Attractiveness Boosted Decision Trees(AdaBoost): Performance x Num Estimators'
plt.title('\n'.join(wrap(title,60)))
plt.plot(max_n_estimators, test_err6, '-', label='test error, max_depth = 6')
plt.plot(max_n_estimators, train_err6, '-', label='train error, max_depth = 6')
plt.plot(max_n_estimators, test_err8, '-', label='test error, max_depth = 8')
plt.plot(max_n_estimators, train_err8, '-', label='train error, max_depth = 8')
plt.plot(max_n_estimators, test_err10, '-', label='test error, max_depth = 10')
plt.plot(max_n_estimators, train_err10, '-', label='train error, max_depth = 10')
plt.legend()
plt.xlabel('Num Estimators')
plt.ylabel('Mean Square Error')
plt.savefig('plots/Attractiveness_Analytics/Boosting/Attractiveness_Analytics_ADABoost_PerformancexNumEstimators.png')
print 'plot complete'



#Boosting_GradientBoostingClassifier
max_n_estimators = range(2, 21, 1)
train_err6 = [0] * len(max_n_estimators)
test_err6 = [0] * len(max_n_estimators)
train_err8 = [0] * len(max_n_estimators)
test_err8 = [0] * len(max_n_estimators)
train_err10 = [0] * len(max_n_estimators)
test_err10 = [0] * len(max_n_estimators)

for i, o in enumerate(max_n_estimators):
    print 'GradientBoostingClassifier: learning a decision tree with n_estimators=' + str(o)
    bdt4 = GradientBoostingClassifier(max_depth=6, n_estimators=o)
    bdt6 = GradientBoostingClassifier(max_depth=8, n_estimators=o)
    bdt8 = GradientBoostingClassifier(max_depth=10, n_estimators=o)
    bdt4.fit(X_train, y_train)
    bdt6.fit(X_train, y_train)
    bdt8.fit(X_train, y_train)
    train_err6[i] = mean_squared_error(y_train,
                                     bdt4.predict(X_train))
    test_err6[i] = mean_squared_error(y_test,
                                    bdt4.predict(X_test))
    train_err8[i] = mean_squared_error(y_train,
                                     bdt6.predict(X_train))
    test_err8[i] = mean_squared_error(y_test,
                                    bdt6.predict(X_test))
    train_err10[i] = mean_squared_error(y_train,
                                     bdt8.predict(X_train))
    test_err10[i] = mean_squared_error(y_test,
                                    bdt8.predict(X_test))
    print '---'

# Plot results
print 'plotting results'
plt.figure()
title = 'Attractiveness Boosted Decision Trees(GradientBoostingClassifier): Performance x Num Estimators'
plt.title('\n'.join(wrap(title,60)))
plt.plot(max_n_estimators, test_err6, '-', label='test error, max_depth = 6')
plt.plot(max_n_estimators, train_err6, '-', label='train error, max_depth = 6')
plt.plot(max_n_estimators, test_err8, '-', label='test error, max_depth = 8')
plt.plot(max_n_estimators, train_err8, '-', label='train error, max_depth = 8')
plt.plot(max_n_estimators, test_err10, '-', label='test error, max_depth = 10')
plt.plot(max_n_estimators, train_err10, '-', label='train error, max_depth = 10')
plt.legend()
plt.xlabel('Num Estimators')
plt.ylabel('Mean Square Error')
plt.savefig('plots/Attractiveness_Analytics/Boosting/Attractiveness_GradientBoostingClassifier_PerformancexNumEstimators.png')
print 'plot complete'
### ---