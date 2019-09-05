# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 00:27:50 2019

@author: Raval
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3] ].values
y = dataset.iloc[:, 4].values

#test,train,split the data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state=0)

#scalling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#fiting the logistic regression to traning set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#predicting the test result
y_pred = classifier.predict(X_test)

#confusion matrix to evalute model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
"""([[65,  3],
       [ 8, 24]]the result will be this means 65+24=89 correct prediction and 8+3=11 is not correct prediction"""

#visualising the training set
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))#meshgrid is for evry pixel in the plot
#start = X_set[:, 0].min() - 1 we take the minimum value of age - 1 because we don't want our value squzzed
#stop = X_set[:, 0].max() + 1 maximum value of the age to get the range of pixles we want include in the frame
#same for the salary
#steps=0.01 means there is 0.2 resolution. if we had choose 0.5 this woudn't have been that dense we would actually see the pixle points.that's better to take this resolution because this arenice prediction regions ,it's continious
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = 'coolwarm')
 

#contourf is for that line sperate
#ravel Return a contiguous flattened array. reffer np.raval documentation.
#plt.xlim we plot x age and y estimated salary to plot age min ,age max,estimated salary min,estimated salary max
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
#np.unique Find the unique elements of an array. refer documentaion
#enumerate iterator for index, value of iterable. refer documentaion
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green' ,'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#visualising the test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

