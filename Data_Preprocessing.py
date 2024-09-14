import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import sklearn


df = pd.read_csv('/home/m-fayzi/Desktop/machin_learning/Data.csv')
# print(df.shape)
x = df.drop(['Purchased'], axis=1).values
y = df['Purchased'].values
# print(y)

# Missing Values 
#Solution 1  For Select data by Dropna
df_dropna = df.copy()
print('Before:', df_dropna.shape)
df_dropna.dropna(inplace=True)
print(df_dropna)
print('After:',df_dropna.shape)


#Solution 2  for select data by Fillna
df_fillna = df.drop(['Country', 'Purchased'],axis=1)
df_fillna.fillna(df_fillna.mean(), inplace=True)
# df_fillna.fillna(df_fillna.median(), inplace=True)
print(df_fillna)
print(df.isnull().sum())
print(df_fillna.isnull().sum())



#Solution 3 use scikit-learn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values =np.nan, strategy='median')
# imputer = SimpleImputer(missing_values =np.nan, strategy='mean')
x[:, 1:3] = imputer.fit_transform(x[:, 1:3])
# imputer.fit(x[:, 1:3])
# x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)



#Solution 1 for null data

#Solution 2 for null data

#Solution 3 for null data


