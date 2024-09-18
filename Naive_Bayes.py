import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import sklearn
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/home/m-fayzi/Desktop/machin_learning/social_network_ads.csv')
# 
# print(df.head())
# print(df.info())
# print(pd.crosstab(df.Gender, df.Purchased))
# print(df['Gender'].value_counts())
# print(df.Gender.value_counts())
# print(df.describe())
X = df.iloc[:, [1,2,3]]
y = df.iloc[:, -1]
# print(X.head())


#LabelEncoder 
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
X['Gender'] = label.fit_transform(X['Gender'])
# print(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0)
# print(X_train.head(1))



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.transform(X_test)
# print(X_train[0])


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

pred_y = classifier.predict(X_test)

# print(pred_y)

#Evaluation Model
from sklearn.metrics import confusion_matrix, accuracy_score
conf_metrix = confusion_matrix(y_test, pred_y)
# print(conf_metrix)
sns.heatmap(conf_metrix, annot=True)
# plt.show()

ac = accuracy_score(y_test, pred_y)
print(ac)