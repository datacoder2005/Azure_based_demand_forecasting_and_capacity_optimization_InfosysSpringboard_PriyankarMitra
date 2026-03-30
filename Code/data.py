import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
start_date = "2022-01-01"
end_date = "2024-12-31"
dates = pd.date_range(start_date, end_date, freq="D")

regions = [
"East US","East US 2","West US","West US 2","West US 3",
"Central US","North Central US","South Central US",
"Canada Central","Canada East","Brazil South",
"North Europe","West Europe","UK South","UK West",
"France Central","Germany West Central","Norway East",
"Switzerland North","UAE North","South Africa North",
"India Central","India South","India West",
"Japan East","Japan West","Korea Central","Korea South",
"Australia East","Australia Southeast",
"Southeast Asia","East Asia"
]

services = ["Compute","Storage"]

rows = []

# Macro drivers (global)
gdp_growth = 0.03 + 0.01*np.sin(2*np.pi*dates.dayofyear/365)
it_spending = np.linspace(100, 125, len(dates))
internet_growth = 0.05 + 0.02*np.sin(2*np.pi*dates.dayofyear/180)

for region in regions:
    region_factor = np.random.uniform(0.8,1.3)
    regional_growth = np.random.uniform(1.05,1.25)

    for service in services:

        base = 4000 if service=="Compute" else 3000
        unit_price = 0.5 if service=="Compute" else 0.2

        capacity = base * 2 * region_factor

        trend = np.linspace(1, regional_growth, len(dates))
        weekly = 1 + 0.08*np.sin(2*np.pi*dates.dayofweek/7)
        noise = np.random.normal(0,0.05,len(dates))

        enterprise_index = 100 + 10*trend

        usage = (
            base * region_factor *
            trend *
            weekly *
            (1+gdp_growth) *
            (1+noise)
        )

        usage = np.minimum(usage, capacity*1.02)

        for i,d in enumerate(dates):

            util = usage[i]/capacity
            headroom = capacity - usage[i]

            incidents = np.random.poisson(0.05)
            availability = 99.95 - incidents*np.random.uniform(0.3,0.8)

            rows.append({
                "timestamp": d,
                "region": region,
                "service_type": service,
                "usage_units": round(usage[i],2),
                "provisioned_capacity": round(capacity,2),
                "utilization_pct": round(util,3),
                "headroom_units": round(headroom,2),
                "unit_price": unit_price,
                "cost_usd": round(usage[i]*unit_price,2),
                "wasted_capacity_cost": round(headroom*unit_price,2),
                "availability_pct": round(availability,3),
                "sla_violation_flag": int(availability<99.9),
                "incident_count": incidents,
                "mttr_minutes": incidents*np.random.randint(20,120),
                "daily_growth_rate": round(trend[i]-trend[i-1],4) if i>0 else 0,
                "weekly_seasonality_index": round(weekly[i],3),
                "spike_flag": int(util>0.9),
                "gdp_growth_rate": round(gdp_growth[i],4),
                "it_spending_index": round(it_spending[i],2),
                "enterprise_demand_index": round(enterprise_index[i],2),
                "internet_traffic_growth": round(internet_growth[i],4)
            })

df = pd.DataFrame(rows)

# -----------------------------
# Inject Dirty Data
# -----------------------------

# Region name variations (without introducing NaNs)

# Create mask once and modify in-place
idx_lower = df.sample(frac=0.02, random_state=42).index
df.loc[idx_lower, "region"] = df.loc[idx_lower, "region"].str.lower()

idx_hyphen = df.sample(frac=0.01, random_state=7).index
df.loc[idx_hyphen, "region"] = df.loc[idx_hyphen, "region"].str.replace(" ", "-", regex=False)

# Final safety check (ensures no NaNs in region)
df["region"] = df["region"].fillna("Unknown")

# Duplicate some rows
df = pd.concat([df, df.sample(frac=0.005)])

df.to_csv("azure_global_demand_dirty_3yr2.csv",index=False)

df.head()
