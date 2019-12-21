# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 23:07:38 2019

@author: Raval
"""

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


#split data into taining and testing
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train= sc_x.fit_transform(X_train) 
X_test= sc_x.transform(X_train)"""