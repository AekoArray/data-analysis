# This is a sample Python script.
import numpy as np
import pandas as pd

from keras.models import Sequential

from keras.layers import Dense, Flatten, Activation

from keras.layers import Dropout

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.utils import np_utils
import tensorflow as tf
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


all_data = pd.read_csv("data.csv", nrows=100)
data = all_data['<OPEN>']

model = tf.keras.Sequential([
    tf.keras.layers.Dense()
])