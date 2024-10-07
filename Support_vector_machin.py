import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import sklearn



df = pd.read_csv('/home/m-fayzi/Desktop/machin_learning/Datasets/mobile_price_range_data.csv')
# print(df.head())
# print(df.info())
# print(df['price_range'].value_counts())
corr = df.corr()
# plt.figure(figsize=(23,18))
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.show()
sort = corr.sort_values(by='price_range', ascending=False).iloc[0].sort_values(ascending=False)
# print(sort)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# print(df.isnull().sum())
from sklearn.preprocessing import StandardScaler
standard = StandardScaler()
X = standard.fit_transform(X)
# print(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 5)
# print(X_train.shape)
# print(X_test.shape)


from sklearn.svm import SVC
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# print(y_pred)
# print(y_test)
# print(pd.crosstab(y_test, y_pred))
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
# print(confusion_matrix(y_test, y_pred))
# print('Accuracy Score:',accuracy_score(y_test, y_pred) * 100)
# print(classification_report(y_test, y_pred))


#SVR Model
data = pd.read_csv('/home/m-fayzi/Desktop/machin_learning/Datasets/Position_Salaries.csv')
# print(data)
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2:3].values

# print(X)
# print(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
scale_y = StandardScaler()
X = scale_X.fit_transform(X)
y = scale_y.fit_transform(y)
print(X)

from sklearn.svm import SVR
model = SVR(kernel='rbf')
model.fit(X,y)

y_pred = model.predict(X)

plt.scatter(X, y, color='magenta')
plt.plot(X, y_pred, color = 'green')
plt.show()