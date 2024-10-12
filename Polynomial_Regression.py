# Import all Needed Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
import numpy as np


#Import datasets csv

df = pd.read_csv('/home/m-fayzi/Desktop/machin_learning/Datasets/FuelConsumption.csv')
# print(df.head())
print(df.info())



# plt.scatter(df.ENGINESIZE, df.CO2EMISSIONS, color = 'red')
# plt.xlabel('ENGINESIZE')
# plt.ylabel('EMISSION')
# plt.show()

X = df[['ENGINESIZE']].values
y = df[['CO2EMISSIONS']].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10, random_state=22)
# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
# print(X_train[:3])
poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(X_train)
# print(train_x_poly)

model = linear_model.LinearRegression()
model.fit(train_x_poly, y_train)
# print('Coeffications', model.coef_)
# print('Intercept', model.intercept_)
test_x_poly = poly.fit_transform(X_test)
y_pred = model.predict(test_x_poly)
# print(y_pred[:3])
# print(y_test[:3])

#Evaluation Model
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))


#Plot
plt.scatter(X_test,y_test, color='red')
plt.plot(X_test, y_pred, color='black')
plt.show()