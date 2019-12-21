# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:29:17 2019

@author: Raval
"""



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10) # cv is cross fold is 10 but we can adjust with 15 and 20
accuracies.mean()
accuracies.std()


# Applying grid search to find best model and best parameter
from sklearn.model_selection import GridSearchCV  
parameters = [{'C' : [1,10,100,1000], 'kernel':['linear']}, # we have to choose whether we choose linear or nonlinear
              {'C' : [1,10,100,1000], 'kernel':['rbf'], 'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}] # gamma is for nonlinear default gamma is auto and auto is 1/N_features so in our dataset we have 2 features so 1/2=0.5 but we there are several parameter that we have to find

# initial gamma parameter we set 'gamma':[0.5, 0.01, 0.1, 0.001, 0.0001] and we get 0.5 so we replace value put values around 0.5
# after getting gamma parameter we set 'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] and we get 0.7             
# rbf is non linear

grid_search = GridSearchCV(estimator=classifier, 
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1) 

grid_search = grid_search.fit(X_train, y_train)

best_acuuracy = grid_search.best_score_ # it give us the all the 10 fold acuuracys mean

best_parameter = grid_search.best_params_ # it gives the best parmeter from above parameters


#  NOTE :- IF WE HAVE A LARGE DATASET THEN WE CAN USE n_jobs=-1 OTHERWISE IT'S OK TO NOT HAVE BUT IF WE USE THIS PARAMETER THEN IT TAKE SOME TIME USE ALL THE POWER OF A MACHINE
     
# parameters contains list of dictionary that tell the optimal no. of parameter to choose
# c is the penalty parameter for svm it heplus to prevent over fitting and grid search tells us that what is the optimal parameter for c between the list [1,10,100,1000] of this



# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
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
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
