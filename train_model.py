import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

X_train = pd.read_csv("data/X_train_scaled.csv").values
y_train = pd.read_csv("data/y_train.csv").values.ravel()

model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=2)

model.save("data/churn_model.h5")
history = model.fit(
    X_train, y_train, 
    epochs=50, batch_size=32, 
    validation_split=0.1, verbose=2
)
import pickle
with open("data/history.pkl", "wb") as f:
    pickle.dump(history.history, f)
