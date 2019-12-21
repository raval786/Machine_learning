# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 22:18:27 2019

@author: Raval
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#.iloc[] is primarily integer position based (from 0 to length-1 of the axis),
# but may also be used with a boolean array
#x is a independent variable
#u is a dependent variable
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)#mean is budefault or its optinal axis=0 is for colum and axis=1 for row
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

X = pd.DataFrame(X)
#encoding labels
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 

labelencoder_X=LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#one hot encodere for dummy variables numerically like in 0 and 1 
#for x
onehotencoder = OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
#for y
labelencoder_Y=LabelEncoder()
y=labelencoder_Y.fit_transform(y)

#split data into taining and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train= sc_x.fit_transform(X_train) 
X_test= sc_x.transform(X_train)