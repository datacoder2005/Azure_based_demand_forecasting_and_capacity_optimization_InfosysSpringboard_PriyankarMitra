import pandas as pd
import numpy as np

df = pd.read_csv("Dataset.csv")

df = df.drop_duplicates()

df["region"] = (
    df["region"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace("-", " ", regex=False)
    .str.replace("_", " ", regex=False)
)

official_regions = {
    "east us": "East US", "east us 2": "East US 2", "west us": "West US", "west us 2": "West US 2", "west us 3": "West US 3",
    "central us": "Central US", "north central us": "North Central US", "south central us": "South Central US",
    "canada central": "Canada Central", "canada east": "Canada East", "brazil south": "Brazil South",
    "north europe": "North Europe", "west europe": "West Europe", "uk south": "UK South", "uk west": "UK West",
    "france central": "France Central", "germany west central": "Germany West Central", "norway east": "Norway East",
    "switzerland north": "Switzerland North", "uae north": "UAE North", "south africa north": "South Africa North",
    "india central": "India Central", "india south": "India South", "india west": "India West",
    "japan east": "Japan East", "japan west": "Japan West", "korea central": "Korea Central", "korea south": "Korea South",
    "australia east": "Australia East", "australia southeast": "Australia Southeast",
    "southeast asia": "Southeast Asia", "east asia": "East Asia"
}

df["region"] = df["region"].map(official_regions)

df = df.dropna(subset=["region"])

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["region", "service_type", "timestamp"])

df["usage_units"] = df.groupby(["region", "service_type"])[
    "usage_units"].transform(lambda x: x.interpolate())
df["availability_pct"] = df.groupby(["region", "service_type"])[
    "availability_pct"].transform(lambda x: x.interpolate())

df["cost_usd"] = df["cost_usd"].fillna(df["usage_units"] * df["unit_price"])

df["utilization_pct"] = df["usage_units"] / df["provisioned_capacity"]
df["headroom_units"] = df["provisioned_capacity"] - df["usage_units"]
df["wasted_capacity_cost"] = df["headroom_units"] * df["unit_price"]

df = df.reset_index(drop=True)

print(df.isnull().sum())
print(df.head())

