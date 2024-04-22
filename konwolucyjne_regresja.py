from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras


# Initializing data
boston = fetch_openml(name="boston", version=1)
print(boston.data)
print("\n")
print(boston.target)


# Plotting heatmap of initialized data
plt.figure()
sns.heatmap(boston.data.corr().abs(), annot=True)
plt.show()


# Defining and merging dataframes
df_data = pd.DataFrame(boston.data)
df_target = pd.DataFrame(boston.target)
df_merged = df_target.join(df_data)
print(df_merged)


# Plotting merged dataframe
plt.figure()
sns.boxplot(df_merged)
plt.show


# Splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target)

# Creating a model
model = keras.models.Sequential()
model.add(keras.layers.Conv1D(30, 2, activation="relu", input_shape=[13, 1]))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(20, activation="relu"))

# Output layer
model.add(keras.layers.Dense(1))

# Model compilation
model.compile(loss="mse", optimizer="adam")

# Model learning
history = model.fit(X_train, y_train, epochs=30)

# Model prediction
y_pred = model.predict(X_test)
