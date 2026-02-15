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
