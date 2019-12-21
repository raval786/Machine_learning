# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 23:25:31 2019

@author: Raval
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#.iloc[] is primarily integer position based (from 0 to length-1 of the axis),
# but may also be used with a boolean array
#x is a independent variable
#y is a dependent variable
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

#using the dendogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendograms = sch.dendrogram(sch.linkage(X, method='ward'))#sch.linkage is algorithm of hierarchical clustering
# method='ward' is we are minimizing the within clusters variance. and that is the variance within each cluseters
plt.title('Dendograms')
plt.xlabel('Customers')
plt.ylabel('Euclidian Distances')
plt.show()

# fitting the hierarchical clustering
from sklearn.cluster import AgglomerativeClustering # bottom to up clustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean' , linkage='ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
