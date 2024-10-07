#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


df = pd.read_csv('/home/m-fayzi/Desktop/machin_learning/Datasets/Mall_Customers.csv')
# print(df.head())
X = df.iloc[:, [3, 4]]. values
# print(X)
print(df.info())

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# print(wcss)

# plt.plot(range(1, 11), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of Cluster')
# plt.ylabel('WCSS')
# plt.show()
# Fit KMeans by n_clusters = 5 
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
kmeans.fit(X)
y_pred = kmeans.predict(X)
print(y_pred)
#Visualising the Cluster
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s = 100, c = 'red', label = 'cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 100, c = 'blue', label = 'cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s = 100, c = 'orange', label = 'cluster 3')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s = 100, c = 'yellow', label = 'cluster 4')
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s = 100, c = 'maroon', label = 'cluster 5')
plt.scatter(kmeans.cluster_centers_[: , 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'black', label= 'Centroids')
plt.title('Clusters Of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1- 100)')
plt.legend()
plt.show()
