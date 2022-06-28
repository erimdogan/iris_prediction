from base64 import encode
from tabnanny import verbose
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("tensorflow_iris/iris.csv")

class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

x = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

encoder = LabelEncoder()
y1 = encoder.fit_transform(y)

Y = pd.get_dummies(y1).values

X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.2, random_state=0)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=64,verbose=2)
"""loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)


deneme_input = np.array([[5.0, 3.6, 5.3, 0.25]])

y_pred = model.predict(deneme_input)
actual = np.argmax(y_test, axis=1)
predicted = np.argmax(y_pred, axis=1)



print("predicted:" + class_names[predicted[0]]+" %"+str(100*np.max(y_pred[0])))
print("actual:" + class_names[actual[0]])"""
model.save("model/iris.h5")
