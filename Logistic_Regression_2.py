# Logistic Regression 

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import sklearn

data = pd.read_csv('/home/m-fayzi/Desktop/machin_learning/iris.csv')

# print(data)
# print(data.head())
# print(data.info())
# sns.pairplot(data, hue='variety')
# plt.show()

# print(data.shape)

# df = data.drop(['variety'], axis=1)
# corr = df.corr()
# print(corr)
# plt.figure(figsize=(12,10))
# sns.heatmap(corr, cmap='coolwarm', annot=True)
# plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

X = data.drop(['variety'], axis=1)
y = data['variety']
 

# print(X.shape)
# print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.2, random_state=5)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)


logReg = LogisticRegression()
logReg.fit(X_train, y_train)

y_pred = logReg.predict(X_test)
# print(y_pred)
print('Accurecy Score:',metrics.accuracy_score(y_test, y_pred))
#Accurecy Score from test_Size 0.4 = 0.983333
#Accurecy Score from test_Size 0.2 = 0.966667