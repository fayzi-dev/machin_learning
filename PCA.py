# Dimensionality Reduction 
# PCA = Principal Component Analysis
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Import datasets
df = pd.read_csv('iris.csv')
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values


# Featur Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

# print(X)


#Applying PCA
pca = PCA(n_components = 2)
X = pca.fit_transform(X)
# print(X)


new_df = pd.DataFrame(data= X, columns=['PCA 1' , 'PCA 2'])
# print(new_df)
# print(pca.components_)
Final_df = pd.concat([new_df, df[['variety']]], axis= 1)
# print(Final_df)

fig = plt.figure(figsize= (8,8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title(' 2 Component PCA', fontsize= 20)
targets = ['Setosa', 'Versicolor', 'Virginica']
color = ['r', 'g', 'b']
for target, color in zip(targets, color):
    indicesTokeep = Final_df['variety'] == target
    ax.scatter(Final_df.loc[indicesTokeep, 'PCA 1'],
               Final_df.loc[indicesTokeep, 'PCA 2'],
               c = color,
               s = 50)


ax.legend(targets)
ax.grid()
plt.show()