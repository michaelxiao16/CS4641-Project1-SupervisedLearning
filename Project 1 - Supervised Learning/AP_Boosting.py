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
from sklearn.model_selection import KFold
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
# df = pd.get_dummies(df)
# X predictors, Y target attribute
print df.head()
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
train_err = [0] * len(max_n_estimators)
test_err = [0] * len(max_n_estimators)

# Use several test depths
max_depths = [2, 4, 6, 8]
for max_depth in max_depths:
    for i, o in enumerate(max_n_estimators):
        print 'AdaBoostClassifier: learning a decision tree with n_estimators=' + str(o) + ' (max_depth ' + str(max_depth) + ')'
        dt = DecisionTreeClassifier(max_depth=max_depth)
        bdt = AdaBoostClassifier(base_estimator=dt, n_estimators=o)
        bdt.fit(X_train, y_train)
        train_err[i] = mean_squared_error(y_train,
                                         bdt.predict(X_train))
        test_err[i] = mean_squared_error(y_test,
                                        bdt.predict(X_test))
        print 'train_err: ' + str(train_err[i])
        print 'test_err: ' + str(test_err[i])
        print '---'

    # Plot results
    print 'plotting results'
    plt.figure()
    title = 'AP Analytics Boosted Decision Trees(AdaBoost, Max Depth = ' + str(max_depth) + '): Performance x Num Estimators'
    plt.title('\n'.join(wrap(title,61)))
    plt.plot(max_n_estimators, test_err, '-', label='test error')
    plt.plot(max_n_estimators, train_err, '-', label='train error')
    plt.legend()
    plt.xlabel('Num Estimators')
    plt.ylabel('Mean Square Error')
    plt.savefig('plots/AP_Analytics/Boosting/APAnalytics_ADABoost' + str(max_depth) + '_PerformancexNumEstimators.png')
    print 'plot complete'
    ### ---


""" Boosting_GradientBoostingClassifier ---------------------------------------------------------------------------- """
max_n_estimators = range(2, 21, 1)
train_err = [0] * len(max_n_estimators)
test_err = [0] * len(max_n_estimators)
max_depths = [5, 6, 7]

for max_depth in max_depths:
    for i, o in enumerate(max_n_estimators):
        print 'GradientBoostingClassifier: learning a decision tree with n_estimators=' + str(o) + ' (max_depth ' + str(max_depth) + ')'
        bdt = GradientBoostingClassifier(max_depth=max_depth, n_estimators=o)
        bdt.fit(X_train, y_train)
        train_err[i] = mean_squared_error(y_train,
                                         bdt.predict(X_train))
        test_err[i] = mean_squared_error(y_test,
                                        bdt.predict(X_test))
        print 'train_err: ' + str(train_err[i])
        print 'test_err: ' + str(test_err[i])
        print '---'

    # Plot results
    print 'plotting results'
    plt.figure()
    title = 'AP Analytics Boosted Decision Trees(GradientBoostingClassifier, Max Depth = ' + str(max_depth) + '): Performance x Num Estimators'
    plt.title('\n'.join(wrap(title,63)))
    plt.plot(max_n_estimators, test_err, '-', label='test error')
    plt.plot(max_n_estimators, train_err, '-', label='train error')
    plt.legend()
    plt.xlabel('Num Estimators')
    plt.ylabel('Mean Square Error')
    plt.savefig('plots/AP_Analytics/Boosting/APAnalytics_GradientBoostingClassifier' + str(max_depth) + '_PerformancexNumEstimators.png')
    print 'plot complete'
    ### ---