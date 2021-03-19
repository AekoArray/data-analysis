import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv("data.csv", nrows=3000, usecols=['<CLOSE>', '<OPEN>', '<HIGH>', '<LOW>'])

print(dataframe.head())

x_train, x_test = train_test_split(dataframe, test_size=0.1)

y_train = x_train.pop('<CLOSE>')
y_test = x_test.pop('<CLOSE>')

model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(x_train, y_train, epochs=5, batch_size=1, verbose=1)
model.save("model1.hdf5")

pred = model.predict(x_test)

print("Предсказанная стоимость:", pred[2], ", правильная стоимость:",y_test.values[2])

#Предсказание на произвольных данных
pred1 = model.predict([[1.11798, 1.1124,  1.1169 ]])

print(pred1[0][0])