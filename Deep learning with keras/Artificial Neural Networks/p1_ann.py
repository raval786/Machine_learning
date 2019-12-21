# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 23:28:29 2019

@author: Raval
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from time import time

#writer = tf.summary.FileWriter("./output",sess.graph)

NAME = "ann"


 
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1]) # for dummy variable
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # deleting first row

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building ANN
import keras
from keras.models import Sequential # to initialize neural neiwork
from keras.layers import Dense # for densily connected neural network

# INittializing the ANN
model = Sequential()

# Adding the input layer and first hidden layer
model.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform")) # units is hidden layer , kernel_initializer is way of assigning the weights  ,  input_dim is input layer
# input_dim =11 we can also write input_shape=(11,) and to re write above

"""model.add(Dense(6,input_shape=(11,), activation="relu")) # input layer
model.add(Dense(6, activation="relu")) # hidden layer
model.add(Dense(1, activation="relu")) # output layer """

# Adding the second layer
model.add(Dense(units = 6 , activation='relu' , kernel_initializer = "uniform"))

# Output layer
model.add(Dense(units = 1 , activation='sigmoid' , kernel_initializer = "uniform")) # we use sigmoid because there is only one neuron but if we have ore then 2  output neuron then we have to use softmax


tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# Compile the ANN
model.compile(optimizer='adam' , loss = 'binary_crossentropy', metrics= ['accuracy']) # compile means we have to apply algorithm like gradient decent to find optimal set of weights in neuron because first we randomly initialize weight and now we have to apply the algorithm to find optimal set of weights

# fitting the training set
model.fit(X_train, y_train ,batch_size =10, epochs = 10, callbacks=[tensorboard])

# Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

model.summary()

