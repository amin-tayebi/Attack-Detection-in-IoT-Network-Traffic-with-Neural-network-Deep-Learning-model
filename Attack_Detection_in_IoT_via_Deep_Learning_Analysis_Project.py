##################################################################################################################################
#I put this just as a simple sample of a Neural Network Model, For having the main codes of the model we can have a collaboration#
#                                                      please contact
##################################################################################################################################
# Import required packages
from keras.models import Sequential
import math
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.utils import to_categorical
# NumPy is a library for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. 
import numpy as np
import matplotlib.pyplot as plt
import os
# Keras is an open-source software library that provides a Python interface for artificial neural networks. 
# Keras acts as an interface for the TensorFlow library. Up until version 2.3, Keras supported multiple backends, including TensorFlow, Microsoft Cognitive Toolkit, Theano, and PlaidML. 
# contains numerous implementations of commonly used neural-network building blocks such as layers, objectives, activation functions, optimizers, and a host of tools to make working with image and text data easier to simplify the coding necessary for writing deep neural network code.
from keras.callbacks import Callback
# pandas is a software library written for the Python programming language for data manipulation and analysis. 
# In particular, it offers data structures and operations for manipulating numerical tables and time series. 
import pandas as pd
# sklearn) is a free software machine learning library for the Python programming language.[3] It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
# Seaborn is an open-source Python library built on top of matplotlib. It is used for data visualization and exploratory data analysis. Seaborn works easily with dataframes and the Pandas library.
import seaborn as sns
from keras.optimizers import SGD
from keras.callbacks import Callback,EarlyStopping
import gzip
#=======================================================================
# Generate dummy training dataset --as 60% of dataset
np.random.seed(2018)
x_train = np.random.random((6000,10))
y_train = np.random.randint(2, size=(6000, 1))

# Generate dummy validation dataset --as 20% of dataset
x_val = np.random.random((2000,10))
y_val = np.random.randint(2, size=(2000, 1))

# Generate dummy test dataset --as 60% of dataset
x_test = np.random.random((2000,10))
y_test = np.random.randint(2, size=(2000, 1))

#Define the model architecture
model = Sequential()
model.add(Dense(512, input_dim=10,activation = "relu")) #Layer 1
model.add(Dense(512,activation = "relu")) #Layer 2
#model.add(Dense(16,activation = "relu")) #Layer 3
#model.add(Dense(8,activation = "relu")) #Layer 4
#model.add(Dense(4,activation = "relu")) #Layer 5
model.add(Dense(1,activation = "sigmoid")) #Output Layer


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
#Explore the data structure using basic python commands
print("Type of the Dataset:",type(y_train))
print("Shape of training data :",x_train.shape)
print("Shape of training labels :",y_train.shape)
print("Shape of testing data :",type(x_test))
print("Shape of testing labels :",y_test.shape)
#Extract the last 100 rows from the training data to create the validation datasets.
x_val = x_train[300:,]
y_val = y_train[300:,]
#Define the model architecture
model = Sequential()
model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
model.add(Dense(6, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])
#Train the model
model.fit(x_train, y_train, batch_size=32, epochs=30, validation_data=(x_val,y_val))

results = model.evaluate(x_test, y_test)
for i in range(len(model.metrics_names)):print(model.metrics_names[i]," : ", results[i])
