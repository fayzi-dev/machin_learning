# Import Datasets

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import sklearn
import sklearn.metrics

# data = pd.read_csv('iris.csv')
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

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn import metrics

# X = data.drop(['variety'], axis=1)
# y = data['variety']
# print(X)
# print(y)
# print(X.shape)
# print(y.shape)
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=5)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# Log_reg = LogisticRegression()
# Log_reg.fit(X_train, y_train)

# y_pred = Log_reg.predict(X_test)
# print(y_pred)
# print('The Accuracy of Logistic Regression is :',metrics.accuracy_score(y_test, y_pred))

data = {
    'y_actual':     [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    'y_predicted':  [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0]
    }

df =pd.DataFrame(data, columns=['y_actual', 'y_predicted'])
confusion_matrix = pd.crosstab(df['y_predicted'], df['y_actual'], rownames=['predicted'], colnames=['actual'])

# print(df)
# print(confusion_matrix)

# sns.heatmap(confusion_matrix, annot=True)
# plt.show()


from sklearn.metrics import confusion_matrix
cunf_metrix = confusion_matrix(df['y_actual'], df['y_predicted'])
# print(cunf_metrix)

#Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(df['y_actual'], df['y_predicted'])
print(accuracy)

#Recall
from sklearn.metrics import recall_score
recall = recall_score(df['y_actual'], df['y_predicted'])
print(recall)


#Precision
from sklearn.metrics import precision_score
precision = precision_score(df['y_actual'], df['y_predicted'])
print(precision)

#F1_Score
from sklearn.metrics import f1_score
f1_score = f1_score(df['y_actual'], df['y_predicted'])
print(f1_score)


#See one Table All Metrics classification
from sklearn.metrics import classification_report
all_metrics = classification_report(df['y_actual'], df['y_predicted'])
print(all_metrics)