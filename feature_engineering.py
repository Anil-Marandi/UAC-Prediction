import pandas as pd

print("Loading cleaned dataset...")

df = pd.read_csv("data/clean_data.csv")

# ----------------------------
# Convert Date
# ----------------------------

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

# ----------------------------
# Target Column
# ----------------------------

target = "Children in HHS Care"

# ----------------------------
# Lag Features
# ----------------------------

df["Lag_1"] = df[target].shift(1)

df["Lag_7"] = df[target].shift(7)

df["Lag_14"] = df[target].shift(14)

# ----------------------------
# Rolling Mean
# ----------------------------

df["RollingMean7"] = df[target].rolling(7).mean()

df["RollingMean14"] = df[target].rolling(14).mean()

# ----------------------------
# Rolling Std
# ----------------------------

df["RollingStd7"] = df[target].rolling(7).std()

# ----------------------------
# Net Pressure
# ----------------------------

df["NetPressure"] = (
    df["Children transferred out of CBP custody"]
    -
    df["Children discharged from HHS Care"]
)

# ----------------------------
# Calendar Features
# ----------------------------

df["Day"] = df["Date"].dt.day

df["Month"] = df["Date"].dt.month

df["Weekday"] = df["Date"].dt.weekday

df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

# ----------------------------
# Remove Missing Rows
# ----------------------------

df = df.dropna()

# ----------------------------
# Save
# ----------------------------

df.to_csv("data/feature_data.csv", index=False)

print("\nFeature Engineering Completed Successfully!")

print(df.head())