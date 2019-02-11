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

""" ------------------------------------------- Data input and formating --------------------------------------------"""
# Read in data
df = pd.read_csv('datasets/AP_Analytics.csv', delimiter=',', quotechar='"')

# Create directories for plots if they don't exist
if(not os.path.exists('plots')):
    dirname = 'plots'
    os.mkdir(dirname)
    dirname = 'AP_Analytics'
    os.mkdir('plots/%s' %dirname)

if(not os.path.exists('plots/AP_Analytics')):
    dirname = 'AP_Analytics'
    os.mkdir('plots/%s' %dirname)
    dirname = 'Boosting'
    os.mkdir('plots/AP_Analytics/%s' %dirname)

if(not os.path.exists('plots/AP_Analytics/Boosting')):
    dirname = 'Boosting'
    os.mkdir('plots/AP_Analytics/%s' %dirname)

# Convert categorical variables into dummy variables
df = pd.get_dummies(df)
# X predictors, Y target attribute
X = df.ix[:, df.columns != 'Chance of Admit ']
y = df['Chance of Admit ']

# Split data into training and testing. Idk about the test size and random_state
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30, random_state=30)
# Standardize features to look like normally distributed data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

""" Boosting with_ADABoost ----------------------------------------------------------------------------------------- """
train_size = len(X_train)
max_n_estimators = range(2, 31, 1)
train_err3 = [0] * len(max_n_estimators)
test_err3 = [0] * len(max_n_estimators)
train_err4 = [0] * len(max_n_estimators)
test_err4 = [0] * len(max_n_estimators)
train_err5 = [0] * len(max_n_estimators)
test_err5 = [0] * len(max_n_estimators)


for i, o in enumerate(max_n_estimators):
    print 'AdaBoostClassifier: learning a decision tree with n_estimators=' + str(o)
    dt4 = DecisionTreeClassifier(max_depth=3)
    dt6 = DecisionTreeClassifier(max_depth=4)
    dt8 = DecisionTreeClassifier(max_depth=5)
    bdt4 = AdaBoostClassifier(base_estimator=dt4, n_estimators=o)
    bdt6 = AdaBoostClassifier(base_estimator=dt6, n_estimators=o)
    bdt8 = AdaBoostClassifier(base_estimator=dt8, n_estimators=o)
    bdt4.fit(X_train, y_train)
    bdt6.fit(X_train, y_train)
    bdt8.fit(X_train, y_train)
    train_err3[i] = mean_squared_error(y_train,
                                     bdt4.predict(X_train))
    test_err3[i] = mean_squared_error(y_test,
                                    bdt4.predict(X_test))
    train_err4[i] = mean_squared_error(y_train,
                                     bdt6.predict(X_train))
    test_err4[i] = mean_squared_error(y_test,
                                    bdt6.predict(X_test))
    train_err5[i] = mean_squared_error(y_train,
                                     bdt8.predict(X_train))
    test_err5[i] = mean_squared_error(y_test,
                                    bdt8.predict(X_test))
    print '---'

# Plot results
print 'plotting results'
plt.figure()
title = 'AP Analytics Boosted Decision Trees(AdaBoost): Performance x Num Estimators'
plt.title('\n'.join(wrap(title,60)))
plt.plot(max_n_estimators, test_err3, '-', label='test error, max_depth = 3')
plt.plot(max_n_estimators, train_err3, '-', label='train error, max_depth = 3')
plt.plot(max_n_estimators, test_err4, '-', label='test error, max_depth = 4')
plt.plot(max_n_estimators, train_err4, '-', label='train error, max_depth = 4')
plt.plot(max_n_estimators, test_err5, '-', label='test error, max_depth = 5')
plt.plot(max_n_estimators, train_err5, '-', label='train error, max_depth = 5')
plt.legend()
plt.xlabel('Num Estimators')
plt.ylabel('Mean Square Error')
plt.savefig('plots/AP_Analytics/Boosting/AP_Analytics_ADABoost_PerformancexNumEstimators.png')
print 'plot complete'
    ### ---


""" Boosting_GradientBoostingClassifier ---------------------------------------------------------------------------- """
max_n_estimators = range(2, 21, 1)
train_err3 = [0] * len(max_n_estimators)
test_err3 = [0] * len(max_n_estimators)
train_err4 = [0] * len(max_n_estimators)
test_err4 = [0] * len(max_n_estimators)
train_err5 = [0] * len(max_n_estimators)
test_err5 = [0] * len(max_n_estimators)


for i, o in enumerate(max_n_estimators):
    print 'GradientBoostingClassifier: learning a decision tree with n_estimators=' + str(o)
    bdt4 = GradientBoostingClassifier(max_depth=3, n_estimators=o)
    bdt6 = GradientBoostingClassifier(max_depth=4, n_estimators=o)
    bdt8 = GradientBoostingClassifier(max_depth=5, n_estimators=o)
    bdt4.fit(X_train, y_train)
    bdt6.fit(X_train, y_train)
    bdt8.fit(X_train, y_train)
    train_err3[i] = mean_squared_error(y_train,
                                     bdt4.predict(X_train))
    test_err3[i] = mean_squared_error(y_test,
                                    bdt4.predict(X_test))
    train_err4[i] = mean_squared_error(y_train,
                                     bdt6.predict(X_train))
    test_err4[i] = mean_squared_error(y_test,
                                    bdt6.predict(X_test))
    train_err5[i] = mean_squared_error(y_train,
                                     bdt8.predict(X_train))
    test_err5[i] = mean_squared_error(y_test,
                                    bdt8.predict(X_test))
    print '---'

# Plot results
print 'plotting results'
plt.figure()
title = 'HR Analytics Boosted Decision Trees(GradientBoostingClassifier): Performance x Num Estimators'
plt.title('\n'.join(wrap(title,60)))
plt.plot(max_n_estimators, test_err3, '-', label='test error, max_depth = 3')
plt.plot(max_n_estimators, train_err3, '-', label='train error, max_depth = 3')
plt.plot(max_n_estimators, test_err4, '-', label='test error, max_depth = 4')
plt.plot(max_n_estimators, train_err4, '-', label='train error, max_depth = 4')
plt.plot(max_n_estimators, test_err5, '-', label='test error, max_depth = 5')
plt.plot(max_n_estimators, train_err5, '-', label='train error, max_depth = 5')
plt.legend()
plt.xlabel('Num Estimators')
plt.ylabel('Mean Square Error')
plt.savefig('plots/AP_Analytics/Boosting/APAnalytics_GradientBoostingClassifier_PerformancexNumEstimators.png')
print 'plot complete'
### ---