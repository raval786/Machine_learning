# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 19:24:53 2019

@author: Raval
"""

# it's a non linear function or polynomial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")

#x is a independent variable
#y is a dependent variable
X = dataset.iloc[:, 1:2].values # x is a form of a vector if we write [:, 1] but if we write [:, 1:2] iit will take as a matrix so the x is matrix and y is a vector
# because in ml x is consider as a feature matrix to process [:, 1:2] is not incuding the second column because the upper bound of a range in python is exclude
# X is a matrix and y is a vector 
y = dataset.iloc[:, 2].values

# fit the linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# fit he polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
#mutiple linear regression
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#visualizing the linear regression
plt.scatter(X,y ,color='Red')
plt.plot(X , lin_reg.predict(X),color= 'Blue')
plt.title("Truth or bluff")
plt.xlabel("Postion leval")
plt.ylabel("Salary")
plt.show()

#visualizing the polynimial regression
plt.scatter(X,y ,color='Red')
plt.plot(X , lin_reg2.predict(poly_reg.fit_transform(X)),color= 'Blue')#we put he poly_reg.fit_transform(X) this instead of poly_reg because
#the poly_reg is already existing so we have to transorm any matrix that transform into the polynomial feature thaats why we use poly_reg.fit_transform(X)
plt.title("Truth or bluff")
plt.xlabel("Postion leval")
plt.ylabel("Salary")
plt.show()
#if we want the better result then increase the degree in polynmial feature

"""# optional
# x_grid is the imagnary data point to get the good curve like we cant get 1.1,4,7,3.9 etc.
X_grid = np.arange(min(X), max(X) , 0.1)
X_grid = X_grid.reshape((len(X_grid),1))# 1 is the column. we reshape because it will give a vector so we have to transform them into the matrix

plt.scatter(X,y ,color='Red')
plt.plot(X_grid , lin_reg2.predict(poly_reg.fit_transform(X_grid)),color= 'Blue')#we put he poly_reg.fit_transform(X) this instead of poly_reg because
#the poly_reg is already existing so we have to transorm any matrix that transform into the polynomial feature thaats why we use poly_reg.fit_transform(X)
plt.title("Truth or bluff")
plt.xlabel("Postion leval")
plt.ylabel("Salary")
plt.show()"""

#predicting the new result with linear regression model
lin_reg.predict(6.5)#we input deirect the leval that we want to see the result

#predicting the new result with polynomial regression model
lin_reg2.predict(poly_reg.fit_transform(6.5))
