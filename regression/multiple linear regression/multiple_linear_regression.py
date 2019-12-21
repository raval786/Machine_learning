# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 23:09:42 2019

@author: Raval
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("50_Startups.csv")

#x is a independent variable
#u is a dependent variable
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


#encodeinf the lables

from sklearn.preprocessing import LabelEncoder , OneHotEncoder
lableencoder_x = LabelEncoder()
X[:,3] = lableencoder_x.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the dummy variable
#delete the first column
X=X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)

#applying multipal linear regression to trainfg set
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_test,y_test)

#predicting the data
y_pred = regression.predict(X_test)

#building the optimal model using the backwardelimination
#np.ones is for formula y=b0+b1*x1+b2*x2
#astye for data type error if we don't write this it's by default array so we have to typecast to int
import statsmodels.formula.api as sm
X=np.append(arr = np.ones((50,1)).astype(int) , values = X , axis = 1 )#axis =0 is for row and axis=1 for column
#endog is dependent variable
#exog is no of observation
X_opt = X[:,[0,1,2,3,4,5]]
regressor_Ols = sm.OLS(endog=y ,exog=X_opt).fit()
#mutilple linear regresion is also called ordinary square language
#p is probability
#sp=0.05 is the significant value that stay in the model the value shoud not be higher than sp value
#the lower the p value the more significant your independent variable is going to be with resect to your independent variable
regressor_Ols.summary()#detail of our model

X_opt = X[:,[0,1,3,4,5]]
regressor_Ols = sm.OLS(endog=y ,exog=X_opt).fit()
regressor_Ols.summary()


X_opt = X[:,[0,3,4,5]]
regressor_Ols = sm.OLS(endog=y ,exog=X_opt).fit() 
regressor_Ols.summary()

X_opt = X[:,[0,3,5]]
regressor_Ols = sm.OLS(endog=y ,exog=X_opt).fit()
regressor_Ols.summary()

X_opt = X[:,[0,5]]
regressor_Ols = sm.OLS(endog=y ,exog=X_opt).fit()
regressor_Ols.summary()

