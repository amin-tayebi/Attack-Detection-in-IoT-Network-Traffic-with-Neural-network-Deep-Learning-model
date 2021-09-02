import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import boston_housing

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

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