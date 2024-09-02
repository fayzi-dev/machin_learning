# Linear Regression :


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

data = pd.read_csv("tips.csv")


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

data.replace({'sex':{'Male': 0 ,'Female': 1 }, 'smoker': {'No': 0 , 'Yes': 1}}, inplace=True)

# print(data.head())
days = pd.get_dummies(data['day'], dtype=int)
data = pd.concat([data, days], axis=1)
print(data.sample(3))

