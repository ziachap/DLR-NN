import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#abalone_train = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv", names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Age"])
ohlc_train = pd.read_csv("C:/backtest-data/SPY.csv", names=["Date", "Open", "High", "Low", "Close"])

ohlc_train.head()
abalone_features = ohlc_train.copy()
abalone_labels = abalone_features.pop('Close')
abalone_features = np.array(abalone_features)

y = np.array([0, 1, 1, 0]);

print(abalone_features)


#Build sequential model
model = Sequential([
    layers.Dense(32, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
]);


model.compile(loss = tf.losses.MeanSquaredError(),
                      optimizer = tf.optimizers.Adam()
                      , metrics=["accuracy"])

model.fit(abalone_features, y,batch_size=2, epochs=20, verbose=1)


print("Hello")

input()
