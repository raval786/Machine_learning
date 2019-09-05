# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 23:09:15 2019

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
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


#split data into taining and testing
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=1/3, random_state=0)

"""#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train= sc_x.fit_transform(X_train) 
X_test= sc_x.transform(X_train)"""


"""from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#imputer = imputer.fit_transform(X[:, 2].values)
X.values[:, [0,1]] = imputer.fit_transform(X.values[:, [0,1]])"""

#simple linear regression 
from sklearn.linear_model import LinearRegression
regresser= LinearRegression()
regresser.fit(X_train,y_train)

#prediction
y_pred = regresser.predict(X_test)

#visualizing the train result
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regresser.predict(X_train) ,color='yellow')
plt.title('salary vs experience')
plt.xlabel('salary')        
plt.ylabel('experience')
plt.show()

#visualizing the test result
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regresser.predict(X_train) ,color = 'blue')
plt.title('salary vs experience')
plt.xlabel('salary')
plt.ylabel('experience')
plt.show()