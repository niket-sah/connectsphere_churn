import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# 1. Load raw data
df = pd.read_csv("data/churn_data.csv")
features = ["call_duration", "data_usage", "contract_length"]
X_all = df[features]

# 2. Load the scaler with joblib, not pandas
scaler = joblib.load("data/scaler.pkl")  
X_scaled = scaler.transform(X_all)

# 3. Load your trained Keras model
model = load_model("data/churn_model.h5")

# 4. Predict churn probabilities
pred_probs = model.predict(X_scaled, batch_size=32, verbose=0).ravel()

# 5. Build churn flag and extract at-risk customers
df["pred_prob"]  = pred_probs
df["pred_churn"] = (pred_probs >= 0.5).astype(int)

at_risk = df.loc[df["pred_churn"] == 1, ["customer_id", "pred_prob"]]
at_risk.to_csv("data/at_risk_customers.csv", index=False)

print(f"Found {len(at_risk)} at-risk customers.")
