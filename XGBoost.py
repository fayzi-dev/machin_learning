
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
#pip install xgboost
import xgboost as xgb

from xgboost import XGBRFClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error,r2_score

"""**Clsddifier model XGBoost**"""

# df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/diabetes.csv")
# df

# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values

# print(X)

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=5)
# print(X_train.shape)

# classifier = XGBRFClassifier()
# classifier.fit(X_train, y_train)

# y_pred = classifier.predict(X_test)
# print(y_pred)

# accuracy = accuracy_score(y_test, y_pred) *100
# print(accuracy)



"""**Regression Model XGBoost**"""

data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/co2.csv")
data.info()

data['Month'] = data.YYYYMM.astype('str').str[4:6].astype(float)
data['Year'] = data.YYYYMM.astype('str').str[0:4].astype(float)

data.info()
data.head(5)

data.drop(["YYYYMM"], axis=1, inplace=True)
# data.info()
print(data.dtypes)

print(data.isnull().sum())

print(data.shape)

X = data.loc[:,['Month', 'Year']].values
y = data.loc[:,'Value'].values
y

data_dmatrix =xgb.DMatrix(X, label=y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

regressor = xgb.XGBRFRegressor(
    n_estimators = 1000,
    learning_rate=0.9,
    subsample=0.75,
    colsample_bytree=1,
    max_depth=7,
    gamma=0,
)
regressor.fit(X_train, y_train)

scores = cross_val_score(regressor, X_train, y_train, cv=10)
print(scores.mean())

prediction = regressor.predict(X_test)
r2 = np.sqrt(r2_score(y_test, prediction))
print(r2)

rmse = np.sqrt(mean_squared_error(y_test, prediction))
print(rmse)

plt.figure(figsize=(10,5), dpi=80)
sns.lineplot(x='Year', y='Value', data=data)

plt.figure(figsize=(10,5), dpi=80)
x_ax =range(len(y_test))
plt.plot(x_ax, y_test, label='test')
plt.plot(x_ax, prediction, label="predicted")
plt.title("Test And Predicted Data")
plt.legend()
plt.show()
