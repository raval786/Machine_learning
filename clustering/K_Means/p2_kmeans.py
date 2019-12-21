# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 23:00:48 2019

@author: Raval
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

#use the elbow method to find the optimal numnber of clusters



from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):#we take 1 to 11 because we take 10 iteration of sum of square method
    kmeans = KMeans(n_clusters= i , init='k-means++' , max_iter=300 , n_init=10 , random_state= 0)#max_iter is maximum no of iteration there can be define to final clusters when kmeans algorithm is runing
    #n_init is no of times kmeans algorithm is run with diffrent initital centroids
    #init we choose kmeans++ because it's powerful and we also choose random but we don't want to fall random initialize fall
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)# inertia_ is compute the cluster sum of square
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('num of cluseters')
plt.ylabel('WCSS')
plt.show()
 

# apply kmeans to dataset
kmeans = KMeans(n_clusters= 5 , init='k-means++' , max_iter=300 , n_init=10 , random_state= 0)#max_iter is maximum no of iteration there can be define to final clusters when kmeans algorithm is runing
y_kmeans=kmeans.fit_predict(X)

#visualize the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1') # X[y_kmeans == 0, 0] is for row and X[y_kmeans == 0, 1] is for column
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')# s is the size
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')# centroids
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()










    
