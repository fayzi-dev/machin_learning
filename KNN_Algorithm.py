import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn



from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
# print(cancer.data.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.25, random_state=42)
# print(X_train.shape)
# print(cancer.target)
# sns.histplot(cancer.target)
# plt.show()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(knn.score(X_train, y_train))
print(knn.score(X_test, y_test))



from sklearn.metrics import confusion_matrix, classification_report
# print(classification_report(y_pred, y_test))

conf_metrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_metrix, annot=True)
plt.show()


# Optimal Value Of K
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_y = knn.predict(X_test)
    error_rate.append(np.mean(pred_y != y_test))


# print(error_rate)

plt.figure(figsize=(10,6))
plt.plot(range(1,40), error_rate, linestyle='dashed', marker='o')
plt.show()



knn = KNeighborsClassifier(n_neighbors= 11)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(knn.score(X_train, y_train))
print(knn.score(X_test, y_test))
cf = confusion_matrix(y_test, y_pred)
sns.heatmap(cf, annot=True)
plt.show()