# Iris prediction
  I tried to classify Iris data set by using a model which I designed simple Neural Network.

## Model
importing required moules
*from base64 import encode
*from tabnanny import verbose
*import tensorflow as tf
*from tensorflow.keras import layers
*import pandas as pd
*import numpy as np
*from tensorflow.keras import datasets, layers, models
*from tensorflow.keras.utils import to_categorical
*from sklearn.model_selection import train_test_split
*from sklearn.preprocessing import LabelEncoder
 

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
