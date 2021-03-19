import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv("data.csv", nrows=1000, usecols=['<CLOSE>', '<OPEN>', '<HIGH>', '<LOW>'])

temp = dataframe.values.tolist()

for i in range(len(temp) - 1):
    temp[i][1], temp[i][2] = temp[i + 1][1], temp[i + 1][2]
temp.pop()

d = {"<OPEN>": [], "<HIGH-1>": [], "<LOW-1>": [], "<CLOSE>": []}
for x in temp:
    d["<OPEN>"].append(x[0])
    d["<HIGH-1>"].append(x[1])
    d["<LOW-1>"].append(x[2])
    d["<CLOSE>"].append(x[3])

dataframe = pd.DataFrame(data=d)

print(dataframe.head())

max_label_value = dataframe['<CLOSE>'].max()
min_label_value = dataframe['<CLOSE>'].min()
d = (max_label_value-min_label_value)/5

dataframe['<CLOSE>'] = dataframe['<CLOSE>'].apply(lambda x: ((x-min_label_value) // d) if not ((x-min_label_value) // d) == 5.0 else ((x-min_label_value) // d) - 1)

print(min_label_value)
print(max_label_value)
print(d)
print(dataframe['<CLOSE>'].max())
dataframe.head()

x_train, x_test = train_test_split(dataframe, test_size=0.1)

y_train = x_train.pop('<CLOSE>')
y_test = x_test.pop('<CLOSE>')
tlabel_train = tf.keras.utils.to_categorical(y_train, num_classes=5)
tlabel_test = tf.keras.utils.to_categorical(y_test, num_classes=5)
print(tlabel_train)

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(3,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(x_train, tlabel_train, epochs=100, batch_size=3, verbose=1)
model.save("model.hdf5")

pred = model.predict(x_test)
