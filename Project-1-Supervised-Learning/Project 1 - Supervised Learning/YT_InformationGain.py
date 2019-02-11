# from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import csv

categories = ['image_id', '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes','Bald', 'Bangs',
              'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
              'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
              'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
              'Receding_Hairline', 'Rosy_Cheeks	', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
              'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

df = pd.read_csv('datasets/Attractiveness_Analytics.csv', delimiter=',', quotechar='"')
df = df.drop('image_id', 1)
df = pd.get_dummies(df)
x = df.ix[:, df.columns != 'Attractive']
y = df['Attractive']

print mutual_info_classif(x, y, discrete_features=True)
res = dict(zip(categories, mutual_info_classif(x, y, discrete_features=True)))
print(res)


with open('YT_InformationGain.csv', 'wb') as output:
    writer = csv.writer(output)
    for key, value in res.iteritems():
        writer.writerow([key, value])


