import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

df = pd.read_csv("../data/forecast_output.csv")

# Only past rows (actual available)
df = df.dropna(subset=["usage_units"])

rmse = np.sqrt(mean_squared_error(df["usage_units"], df["forecast"]))

print(f"RMSE: {rmse:.2f}")

# Threshold
if rmse > 150:
    print("🚨 ALERT: Model performance degrading!")
else:
    print("✅ Model performing well")