# Linear Regression :


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# data = pd.read_csv("tips.csv")


# print(data.head())
# print(data.info())
# print(data.shape)
# print(data.describe())
# print(data.sample(10))
# print(data.groupby("day").count())

# df2 = data.groupby('day').sum()
# df2.drop(['smoker','sex','time'], inplace=True, axis=1)
# df2['percent'] = df2['tip'] / df2['total_bill'] * 100
# print(df2)



# df3 = data.groupby('smoker').sum()
# df3['percent'] =df3['tip'] / df3['total_bill'] * 100
# print(df3)



# df4 = data.groupby(['day','size']).sum()
# df4['percent'] = df4['tip'] / df4['total_bill'] * 100
# print(df4)


# sns.catplot(x='day', kind='count', data=data, color='green')
# plt.show()

# sns.catplot(x='day', hue='size', kind='count', data=data)
# plt.show()

# data.replace({'sex':{'Male': 0 ,'Female': 1 }, 'smoker': {'No': 0 , 'Yes': 1}}, inplace=True)

# print(data.head())
# days = pd.get_dummies(data['day'], dtype=int)

# print(days.sample(10)

# data = pd.concat([data, days], axis=1)

# times = pd.get_dummies(data['time'], dtype=int)
# data = pd.concat([data, times], axis=1)

# print(data.sample(3))
# X = data[['sex', 'smoker', 'size', 'Fri', 'Sat', 'Sun', 'Dinner']]
# print(X)
# y = data[['tip']]
# print(Y)

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn import metrics


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 26)

# print(X_train)
# print(y_train)

# reg = LinearRegression()
# reg.fit(X_train, y_train)

# predict = reg.predict(X_test)
# print(predict)
# print(y_test)

# sns.displot(y_test-predict)
# g = sns.FacetGrid(data, col='time', row='sex')
# g = sns.FacetGrid(data, col='time', hue='sex')
# g = sns.FacetGrid(data, col='time')
# g.map(sns.scatterplot, 'total_bill', 'tip')
# g.map(sns.scatterplot, 'total_bill','tip','sex')
# g.map_dataframe(sns.scatterplot, x='total_bill',y='tip',hue='sex')
# g.add_legend()
# sns.relplot(data=data, x='total_bill', y='tip', hue='day', col='time', row='sex', style='time')
# sns.relplot(data=data, x='total_bill', y='tip', hue='smoker', style='time')
# sns.relplot(data=data, x='total_bill', y='tip', hue='smoker', style='time', kind='line')
# sns.relplot(data=data, x='total_bill', y='tip', size='size')
# sns.scatterplot(data=data, x='total_bill', y='tip', hue='size', size='size',
#                 sizes=(20, 200), legend='full'
#                 )


# sns.regplot(data=data, x='total_bill', y='tip')
# pl = sns.PairGrid(data)
# pl.map(sns.scatterplot)
# plt.show()

# print('Mean aboulute error:', metrics.mean_absolute_error(y_test,predict))
# print('Mean squared error:', metrics.mean_squared_error(y_test,predict))
# print('Root Mean squared error:', np.sqrt(metrics.mean_squared_error(y_test,predict)))

# New Data
# print(X.head())

# new_customer = np.array([0,1,2,1,0,0,1]).reshape(1, -1)
# new_customer = np.array([0,1,3,1,0,0,0]).reshape(1, -1)
# print(new_customer)
# new_customer_Predict = reg.predict(new_customer)
# print(new_customer_Predict)


data = pd.read_csv('boston.csv')
data = data.rename(columns={'medv':'price'})

# print(data.head())
# print(data.shape)
# print(data.columns)
# print(data.info())
# print(data.nunique())
# print(data.isnull().sum())
# print(data.describe())



# def select_data (df, lower ,upper):
#     mask = (df < lower | df > upper)
#     selected_data = df[mask]
#     return select_data

# corr = select_data()


# corr = data.corr()
# max_corr =corr[corr >= 0.5]
# max_corr =corr[corr <= -0.5]

# print(corr.shape)
# plt.figure(figsize=(10, 10))
# plt.figure(figsize=(15, 8))
# sns.heatmap(max_corr, annot=True,cmap='coolwarm', cbar=True)
# sns.heatmap(corr, annot=True,cmap='coolwarm', cbar=True)
# plt.show()


# X = data[['zn']]
X = data[['rm']]
y = data['price']

# print(X)
# print(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 4)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

from sklearn.linear_model import LinearRegression

slr = LinearRegression()
slr.fit(X_train, y_train)

# print(slr.intercept_)
# print(slr.coef_)

from sklearn import metrics

y_pred = slr.predict(X_train)

print('R^2:',metrics.r2_score(y_train, y_pred))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))


sns.scatterplot(x=y_train, y=y_pred)
# plt.scatter(y_train, y_pred)
plt.show()

sns.displot(y_train-y_pred)
plt.show()

