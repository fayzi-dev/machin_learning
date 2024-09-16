import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import sklearn

# Cross_validation

dataset = pd.read_csv('/home/m-fayzi/Desktop/machin_learning/social_network_ads.csv')
dataset.drop(['User ID', 'Gender'], axis=1, inplace=True)
# print(dataset)
# print(dataset.columns)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# print(X)
# print(y)


# Train and Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2 ,random_state=1)
# print(X_train)
#Standard Scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.transform(X_test)

#Training the kernel SVM model on the Training set
from sklearn.svm import SVC

classifier = SVC(kernel='rbf', random_state = 0)
print(classifier.fit(x_train, y_train))

#Making The Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

Accuracy_Score = accuracy_score(y_test, y_pred)
print('Accuraxy Befor k-Fold:',Accuracy_Score)



# Appling k-Fold Cross Validation

from sklearn.model_selection import cross_val_score
accurecies = cross_val_score(estimator=classifier, X= x_train, y= y_train, cv = 10)
# print(accurecies)
print('Accuracy After k-Fold:',accurecies.mean() * 100 )
print('Standard Deviation:',accurecies.std() * 100 )