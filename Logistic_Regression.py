# Import Datasets

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import sklearn

data = pd.read_csv('iris.csv')
# print(data.head())
# print(data['variety'].value_counts())
# print(data.info())
# sns.pairplot(data, hue='variety', markers='+')
# print(plt.show())
# print(data.shape)

# corr = data.drop(['variety'], axis=1).corr()
# corr = data.corr()
# plt.figure(figsize=(10,8))
# sns.heatmap(corr, cmap='viridis', annot=True)
# plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

X = data.drop(['variety'], axis=1)
y = data['variety']
# print(X)
# print(y)
# print(X.shape)
# print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=5)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

Log_reg = LogisticRegression()
Log_reg.fit(X_train, y_train)

y_pred = Log_reg.predict(X_test)
print(y_pred)
print('The Accuracy of Logistic Regression is :',metrics.accuracy_score(y_test, y_pred))