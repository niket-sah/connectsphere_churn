import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, classification_report

X_test  = pd.read_csv("data/X_test_scaled.csv").values
y_test  = pd.read_csv("data/y_test.csv").values.ravel()
model   = load_model("data/churn_model.h5")

y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
