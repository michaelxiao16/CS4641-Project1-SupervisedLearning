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


df = pd.read_csv('datasets/Attractiveness.csv', delimiter=',', quotechar='"')

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

print(df.head(5))
df = df.drop(df.index[10000:])

df.to_csv('datasets/Attractiveness_Analytics.csv')