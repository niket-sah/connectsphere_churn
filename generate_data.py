import pandas as pd
import numpy as np

np.random.seed(42)
n = 200
df = pd.DataFrame({
    "customer_id": [f"C{idx:04d}" for idx in range(1, n+1)],
    "call_duration": np.random.randint(50, 501, n),
    "data_usage": np.round(np.random.uniform(1.0, 50.0, n), 2),
    "contract_length": np.random.randint(1, 37, n),
    "churn": np.random.choice([0,1], n, p=[0.8,0.2])
})
df.to_csv("data/churn_data.csv", index=False)
