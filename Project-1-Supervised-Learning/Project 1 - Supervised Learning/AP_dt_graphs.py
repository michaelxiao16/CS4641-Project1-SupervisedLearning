import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import os


df = pd.read_csv('datasets/AP_Analytics.csv', delimiter=',', quotechar='"')

if(not os.path.exists('plots')):
    dirname = 'plots'
    os.mkdir(dirname)
    dirname = 'AP_Analytics'
    os.mkdir('plots/%s' %dirname)

if(not os.path.exists('plots/AP_Analytics')):
    dirname = 'AP_Analytics'
    os.mkdir('plots/%s' %dirname)
    dirname = 'DecisionTree'
    os.mkdir('plots/AP_Analytics/%s' %dirname)

if(not os.path.exists('plots/AP_Analytics/DecisionTree')):
    dirname = 'DecisionTree'
    os.mkdir('plots/AP_Analytics/%s' %dirname)


df = pd.get_dummies(df)
X = df.ix[:, df.columns != 'Chance of Admit ']
y = df['Chance of Admit ']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30, random_state=30)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#DecisionTreeClassifier
train_size = len(X_train)
offsets = range(int(0.1 * train_size), train_size, int(0.05 * train_size))
train_err3 = [0] * len(offsets)
test_err3 = [0] * len(offsets)
train_err4 = [0] * len(offsets)
test_err4 = [0] * len(offsets)
train_err5 = [0] * len(offsets)
test_err5 = [0] * len(offsets)

print 'training_set_max_size:', train_size, '\n'
for i, o in enumerate(offsets):
    print 'learning a decision tree with training_set_size=' + str(o)
    clf = DecisionTreeClassifier(max_depth=3)
    X_train_temp = X_train[:o].copy()
    y_train_temp = y_train[:o].copy()
    X_test_temp = X_test[:o].copy()
    y_test_temp = y_test[:o].copy()
    clf = clf.fit(X_train_temp, y_train_temp)

    train_err3[i] = mean_squared_error(y_train_temp,
            clf.predict(X_train_temp))
    test_err3[i] = mean_squared_error(y_test_temp,
                                     clf.predict(X_test_temp))
    print 'train_err: ' + str(train_err3[i])
    print 'test_err: ' + str(test_err3[i])
    print '---'
for i, o in enumerate(offsets):
    print 'learning a decision tree with training_set_size=' + str(o)
    clf = DecisionTreeClassifier(max_depth=4)
    X_train_temp = X_train[:o].copy()
    y_train_temp = y_train[:o].copy()
    X_test_temp = X_test[:o].copy()
    y_test_temp = y_test[:o].copy()
    clf = clf.fit(X_train_temp, y_train_temp)

    train_err4[i] = mean_squared_error(y_train_temp,
            clf.predict(X_train_temp))
    test_err4[i] = mean_squared_error(y_test_temp,
                                     clf.predict(X_test_temp))
    print 'train_err: ' + str(train_err4[i])
    print 'test_err: ' + str(test_err4[i])
    print '---'
for i, o in enumerate(offsets):
    print 'learning a decision tree with training_set_size=' + str(o)
    clf = DecisionTreeClassifier(max_depth=5)
    X_train_temp = X_train[:o].copy()
    y_train_temp = y_train[:o].copy()
    X_test_temp = X_test[:o].copy()
    y_test_temp = y_test[:o].copy()
    clf = clf.fit(X_train_temp, y_train_temp)

    train_err5[i] = mean_squared_error(y_train_temp,
            clf.predict(X_train_temp))
    test_err5[i] = mean_squared_error(y_test_temp,
                                     clf.predict(X_test_temp))
    print 'train_err: ' + str(train_err5[i])
    print 'test_err: ' + str(test_err5[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
plt.title('Decision Trees: Performance x Training Set Size for Max Depths')
plt.plot(offsets, test_err3, '-', label='test error, max_depth = 3')
plt.plot(offsets, train_err3, '-', label='train error, max_depth = 3')
plt.plot(offsets, test_err4, '-', label='test error, max_depth = 4')
plt.plot(offsets, train_err4, '-', label='train error, max_depth = 4')
plt.plot(offsets, test_err5, '-', label='test error, max_depth = 5')
plt.plot(offsets, train_err5, '-', label='train error, max_depth = 5')
plt.legend()
plt.xlabel('Training Set Size')
plt.ylabel('Mean Square Error')
filename = 'APAnalytics_DT_PerformancexTrainingSetSize_MAXDEPTHcombo.png'
plt.savefig('plots/AP_Analytics/DecisionTree/%s' % filename)
print 'plot complete'