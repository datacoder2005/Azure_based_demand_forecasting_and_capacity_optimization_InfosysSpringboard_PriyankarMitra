import pandas as pd
import joblib
import json

# =========================
# LOAD MODEL
# =========================
model = joblib.load("E:/College/infosys springboard project/model/xgb_model.pkl")

# Load feature list
with open("E:/College/infosys springboard project/model/feature_cols.json", "r") as f:
    feature_cols = json.load(f)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("E:/College/infosys springboard project/data/processed_data.csv")

# =========================
# PREPROCESS (IMPORTANT)
# =========================

# One-hot encoding SAME as training
df_ml = pd.get_dummies(df, columns=["region", "service_type"], drop_first=True)

# Add missing columns
for col in feature_cols:
    if col not in df_ml.columns:
        df_ml[col] = 0

# Keep only required columns
X = df_ml[feature_cols]

# =========================
# PREDICT
# =========================
df["forecast"] = model.predict(X)

# =========================
# SAVE
# =========================
df.to_csv("E:/College/infosys springboard project/data/forecast_output.csv", index=False)

print("✅ Forecast file created successfully")