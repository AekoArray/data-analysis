import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

train_file_path = 'eurusd.csv'

dataframe = pd.read_csv(train_file_path,sep=';')

max_label_value=dataframe["C(t)"].max()
min_label_value=dataframe["C(t)"].min()
d=(max_label_value-min_label_value)/5

dataframe["C(t)"]=dataframe["C(t)"].apply(lambda x: ((x-min_label_value)//d ))

print(min_label_value)
print(max_label_value)
print(d)
print(dataframe["C(t)"].max())
dataframe.head()

x_train, x_test = train_test_split(dataframe, test_size=0.1)


y_train = x_train.pop('C(t)')
y_test = x_test.pop('C(t)')
tlabel_train = tf.keras.utils.to_categorical(y_train, num_classes=5)
tlabel_test = tf.keras.utils.to_categorical(y_test, num_classes=5)
print(tlabel_train)


model = Sequential()
model.add(Dense(10, activation='relu',input_shape=(3,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(x_train, tlabel_train, epochs=100, batch_size=3, verbose=1)
