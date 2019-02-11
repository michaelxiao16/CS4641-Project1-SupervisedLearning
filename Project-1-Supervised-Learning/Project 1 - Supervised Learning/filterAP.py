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


df = pd.read_csv('datasets/Admission_Predict.csv', delimiter=',', quotechar='"')

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


# mapping = {'low': 1, 'medium': 2, 'high': 3}
# df.replace({'salary': mapping})
# print(df.head(5))
# df = pd.get_dummies(df)
# print(df.head(5))


df['Chance of Admit '] = pd.cut(
    df['Chance of Admit '],
    [0, .20, .40, .60, .80, 1.00],
    labels = [1,2,3,4,5]
    #labels=["very low", "low", "neutral", "high", "very high"]
)
# df = pd.get_dummies(df)
print(df.head(5))

df.to_csv('datasets/AP_Analytics.csv', index=False)