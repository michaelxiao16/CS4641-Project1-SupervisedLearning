# from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

categories = ['GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research',]
df = pd.read_csv('datasets/AP_Analytics.csv', delimiter=',', quotechar='"')
df = pd.get_dummies(df)
df = df.drop('Serial No.', 1)
x = df.ix[:, df.columns != 'Chance of Admit ']
y = df['Chance of Admit ']

print mutual_info_classif(x, y, discrete_features=True)
res = dict(zip(categories, mutual_info_classif(x, y, discrete_features=True)))
print(res)
