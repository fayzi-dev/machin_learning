import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

df = pd.read_csv('/home/m-fayzi/Desktop/machin_learning/Datasets/social_network_ads.csv')

# X = df.iloc[:, 2:4].values
# y = df.iloc[:, 4].values

# print(X,y)

from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
# print(X_train, X_test, y_train, y_test)

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
# X_test = scale.fit_transform(X_test)
# X_train = scale.fit_transform(X_train)
# print(X_train)


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
# classifier.fit(X_train,y_train)
# y_pred = classifier.predict(X_test)

# print(pd.crosstab(y_test, y_pred))
from sklearn.metrics import confusion_matrix, accuracy_score
# print(confusion_matrix(y_test, y_pred))
# print(accuracy_score(y_test, y_pred) * 100)


# Tets model
# print(classifier.predict(scale.transform([[30,87000]]))) #Output : [0] not buy


###Decision tree Regression

data = pd.read_csv('/home/m-fayzi/Desktop/machin_learning/Datasets/Position_Salaries.csv')
# print(data.info())


X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values
# print(X, y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
# print(X_train, y_train, X_test, y_test)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)
predict_y = regressor.predict(X_test)
# print(predict_y)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# print(confusion_matrix(y_test,predict_y))
print(accuracy_score(y_test,predict_y))
# print(classification_report(y_test,predict_y))

## For example full Errors
