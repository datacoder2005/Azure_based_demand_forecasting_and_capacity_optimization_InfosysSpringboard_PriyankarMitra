import pandas as pd
import joblib
import json
from xgboost import XGBRegressor

df = pd.read_csv("../data/processed_data.csv")

with open("../model/feature_cols.json") as f:
    feature_cols = json.load(f)

# Same preprocessing
df_ml = pd.get_dummies(df, columns=["region", "service_type"], drop_first=True)

# Align columns
for col in feature_cols:
    if col not in df_ml.columns:
        df_ml[col] = 0

X = df_ml[feature_cols]
y = df["usage_units"]

model = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.1)
model.fit(X, y)

joblib.dump(model, "../model/xgb_model.pkl")

print("✅ Model retrained successfully")