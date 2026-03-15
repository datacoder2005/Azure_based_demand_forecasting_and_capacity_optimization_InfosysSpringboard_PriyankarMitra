# Azure Demand Forecasting & Capacity Optimization System

## Project Overview

This project focuses on building a predictive system to forecast Azure Compute and Storage demand across global regions. The objective is to support data-driven capacity provisioning decisions, reduce over- and under-investment in infrastructure, and improve overall forecasting accuracy.


## Milestone 1 — Data Collection & Preparation (Completed)

### Data Collection

- Generated 3 years (2022–2024) of daily Azure demand data across global regions for Compute and Storage services.  
- Incorporated operational metrics such as utilization, headroom, availability, incident counts, and SLA indicators.  
- Integrated external macroeconomic drivers including GDP growth rate, IT spending index, enterprise demand index, and internet traffic growth.  

### Data Cleaning & Validation

- Standardized region names and removed duplicate records to ensure categorical consistency.  
- Handled missing values using time-series interpolation and recalculated derived cost metrics.  
- Validated metric relationships (utilization, headroom, wasted capacity cost) to ensure data integrity and model readiness.  


## Milestone 2 — Feature Engineering & Exploratory Analysis (Completed)

### Feature Engineering

- Created time-based features (year, month, weekday, quarter, day-of-year) to capture temporal demand patterns.  
- Engineered lag features (lag_1, lag_7, lag_30) to model short-term memory and weekly/monthly demand cycles.  
- Computed rolling statistics (7-day and 30-day rolling mean and standard deviation) to capture trend and volatility.  
- Developed percentile-based capacity indicators (capacity risk flag, underutilization flag) and a continuous utilization gap metric for decision support.  

### Statistical Tests & Model Justification

- Conducted stationarity testing (ADF test) across all 64 region-service time series, confirming majority non-stationary behavior and the need for differencing-based models.  
- Performed seasonal strength analysis, identifying strong weekly seasonality across regions.  
- Evaluated autocorrelation at lags 1, 7, and 30, confirming dominant weekly dependency patterns.  
- Analyzed macro-variable correlations and multicollinearity to select optimal features and modeling strategy.  

These analyses provide statistical justification for using SARIMA (with seasonal period = 7) and gradient boosting models (e.g., XGBoost) in the next milestone.

## Milestone 3 — Model Development & Forecasting 

### Time Series Forecasting Models

- Implemented classical time-series models to capture region-specific demand dynamics for each region–service pair.
- Developed ARIMA models for each of the 64 region-service time series to model non-seasonal demand patterns using optimized (p, d, q) parameters.
- Implemented SARIMA models to capture both trend and weekly seasonal patterns using seasonal order (P, D, Q, 7).
- Performed hyperparameter tuning using grid search with RMSE evaluation on a hold-out test set to identify optimal model parameters.

### Machine Learning Forecasting Model

- Built a global machine learning forecasting model capable of learning cross-region demand patterns.
- Implemented XGBoost regression model using engineered temporal, lag, rolling, and macroeconomic features.
- Encoded categorical variables (region and service_type) using one-hot encoding to allow the model to learn regional demand variations.
- Conducted hyperparameter tuning using GridSearchCV with TimeSeriesSplit to optimize model parameters (number of trees, depth, learning rate).

### Model Evaluation & Forecast Generation

- Generated daily forecasts for the test period across all regions and services.
- Computed evaluation metrics including MAE, RMSE, and prediction bias to compare model performance.
- Consolidated predictions from ARIMA, SARIMA, and XGBoost into a unified forecast table containing timestamp, region, service type, actual demand,      predicted demand, and model name.

This forecasting framework provides multiple modeling approaches for demand estimation and forms the foundation for capacity optimization and dashboard visualization in the next milestone.
