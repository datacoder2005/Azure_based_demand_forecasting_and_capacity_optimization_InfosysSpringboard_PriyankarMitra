# 🚀 Azure Demand Forecasting & Capacity Optimization System

## 📌 Project Overview

This project builds an end-to-end predictive system to forecast Azure Compute and Storage demand across global regions.
The goal is to enable **data-driven capacity planning**, reduce **over/under-provisioning**, and improve **forecast accuracy** using statistical and machine learning models.

---

## 🏗️ Project Architecture

* Data Collection & Preprocessing
* Feature Engineering (Time + Lag + Rolling + Macro Features)
* Forecasting Models:

  * ARIMA / SARIMA (Time Series)
  * XGBoost (Machine Learning)
* Batch Prediction Pipeline
* Streamlit Dashboard
* FastAPI Deployment
* Scheduler (Automation)
* Monitoring (Model Performance)
* Retraining Pipeline

---

## 📂 Folder Structure

```
api/           → FastAPI deployment  
batch/         → Batch prediction scripts  
dashboard/     → Streamlit dashboard  
data/          → Processed & forecast data  
model/         → Trained model & feature columns  
scheduler/     → Daily automation script  
monitoring/    → Model performance tracking  
retraining/    → Model retraining pipeline  
documentation/ → Reports / project docs  
```

---

## 📊 Milestone 1 — Data Collection & Preparation ✅

### 🔹 Data Collection

* Generated 3 years (2022–2024) of daily Azure demand data across regions
* Included operational metrics (utilization, headroom, availability, incidents, SLA)
* Integrated macroeconomic drivers (GDP growth, IT spending, enterprise demand, internet traffic growth)

### 🔹 Data Cleaning & Validation

* Standardized categories and removed duplicates
* Handled missing values using interpolation
* Validated metric relationships for consistency

---

## 📈 Milestone 2 — Feature Engineering & EDA ✅

### 🔹 Feature Engineering

* Time features: month, weekday, quarter
* Lag features: lag_1, lag_7, lag_30
* Rolling stats: rolling_mean_7, rolling_std_7
* Capacity indicators: risk flag, utilization gap

### 🔹 Statistical Analysis

* ADF Test → non-stationarity confirmed
* Seasonality → strong weekly pattern
* ACF → strong lag dependencies (1, 7, 30)
* Correlation & multicollinearity analysis

👉 Justifies use of **SARIMA + XGBoost**

---

## 🤖 Milestone 3 — Model Development & Forecasting ✅

### 🔹 Time Series Models

* ARIMA for trend modeling
* SARIMA for seasonal patterns
* Hyperparameter tuning using grid search

### 🔹 Machine Learning Model

* XGBoost Regressor
* Uses temporal + lag + rolling + macro features
* One-hot encoding for region & service
* Tuned using TimeSeriesSplit

### 🔹 Evaluation

* Metrics: MAE, RMSE, Bias
* Combined predictions into unified dataset

---

## 🚀 Milestone 4 — Deployment & Automation ✅

### 🔹 Dashboard (Streamlit)

* Interactive filters (region, service, year)
* Actual vs Forecast visualization
* 30-day future prediction
* Capacity insights

### 🔹 API (FastAPI)

* `/predict` endpoint for real-time inference

### 🔹 Batch Pipeline

* Generates forecast_output.csv using trained model

### 🔹 Scheduler

* Automates daily prediction

### 🔹 Monitoring

* Tracks model performance (RMSE)

### 🔹 Retraining

* Supports future updates

---

## ⚙️ How to Run the Project

### 🔹 1. Install Dependencies

```bash
pip install -r requirement.txt
```

---

### 🔹 2. Run Dashboard

```bash
streamlit run dashboard/milestone4_dashboard.py
```

---

### 🔹 3. Run API

```bash
cd api
uvicorn main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000/docs
```

---

### 🔹 4. Run Batch Prediction

```bash
python batch/batch_predict.py
```

---

### 🔹 5. Run Scheduler

```bash
python scheduler/run_daily.py
```

---

## 📊 Key Features

* Multi-model forecasting (ARIMA, SARIMA, XGBoost)
* Region-level demand prediction
* Real-time API inference
* Automated pipeline
* Interactive dashboard
* 30-day future forecasting

---

## 🎯 Business Impact

* Improves infrastructure planning
* Reduces over-provisioning costs
* Identifies capacity risks early
* Enables data-driven decision making

---

## 📜 License

MIT License
