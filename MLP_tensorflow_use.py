import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pydot
import numpy as np


# Data load
fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()


# Show data
# plt.imshow(x_train_full[8], cmap="gray", vmin=0, vmax=255)
# plt.show()


# Creating list of classes
class_names = [
    "Tshit",
    "Spodnie",
    "sweter",
    "sukienka",
    "plaszcz",
    "sandal",
    "koszula",
    "tenisowka",
    "torebka",
    "but",
]


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))

# Output layer
model.add(keras.layers.Dense(10, activation="softmax"))

# model.compile(
#     loss="sparse_categorical_crossentropy", optimizer="sgd", metrics="accuracy"
# )


# Learning
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd")
model.fit(x_train_full, y_train_full, epochs=30)

# Prediction
y = model.predict(x_test)
print(y[0])

# Showing which category our sample is
category = np.argmax(y, axis=1)
print(category[0])


a = model.evaluate(x_test, y_test)
print(a[0])  # Loss
print(a[1])  # Accuracy
