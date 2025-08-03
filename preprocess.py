import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("data/churn_data.csv")
X = df[["call_duration","data_usage","contract_length"]]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

joblib.dump(scaler, "data/scaler.pkl")
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv("data/X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled,  columns=X.columns).to_csv("data/X_test_scaled.csv",  index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv",   index=False)
