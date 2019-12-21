# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 17:45:51 2019

@author: Raval
"""

# Importing keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D 
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialize the Cnn
model = Sequential()

# step 1 Convolutional
model.add(Convolution2D(32 ,(3, 3), input_shape = (64, 64, 3), activation='relu')) # 32 is thr feature detector. (3,3) is the stride size . input_shape = (64, 3, 3) 64 is we select 64 out of 255 colour density because if choose 128 or 255 it tales hours to train. 3 is the colour channel red,green,blue(rgb) 
# relu is non linear fuction and image is also non-linear so take nonlinearity we choose relu.

# step 2 Pooling
model.add(MaxPooling2D(pool_size = (2, 2))) # it reduce the size of the feature map

# Adding a second Convolutional layer
model.add(Convolution2D(32 ,(3, 3), activation='relu')) #  we don't use input_shape because we use input_shape first to initialize and this layer take automatically the prevoious layer 
# we can choose 64 or 128 instead of 32 to increase the accuracy
model.add(MaxPooling2D(pool_size = (2, 2))) 

# Flattening
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu')) # hidden layer
model.add(Dense(1, activation='sigmoid')) # output layer

# Compile the model
model.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # loss = 'binary_crossentropy' because we want dog or cat but if want more output like dog,cat,bird,cow like more then two outcome so that time we have to choose categorical_crossentropy

# Fitting the CNN model

# took code from keras documentaion website
# Docs » Preprocessing » Image Preprocessing

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255, # it converts the 255 scale image batween 0 to 1.
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64), # 64 is from input_shape
                                                    batch_size=32,
                                                    class_mode='binary')

                                                    
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')
        

model.fit_generator(training_set,
                    steps_per_epoch=2000,
                    epochs=20,
                    validation_data=test_set,
                    validation_steps=400)
       
        
