import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("data/HHS_Unaccompanied_Alien_Children_Program.csv")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMNS ==========")
print(df.columns)

# ----------------------------
# Convert Date Column
# ----------------------------

df["Date"] = pd.to_datetime(df["Date"])

# Sort by Date
df = df.sort_values("Date")

# Set Date as Index
df.set_index("Date", inplace=True)

# ----------------------------
# Convert all remaining columns to numeric
# ----------------------------

for col in df.columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------------------------
# Missing Values
# ----------------------------

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Fill missing values safely
df = df.ffill()
df = df.bfill()

# ----------------------------
# Remove duplicates
# ----------------------------

df = df.drop_duplicates()

# ----------------------------
# Save Clean Dataset
# ----------------------------

df.to_csv("data/clean_data.csv")

print("\nCleaned dataset saved successfully!")

# ----------------------------
# Dataset Info
# ----------------------------

print("\n========== INFO ==========")
print(df.info())

print("\n========== STATISTICS ==========")
print(df.describe())

# ----------------------------
# Plot Target Variable
# ----------------------------

plt.figure(figsize=(15,6))

plt.plot(
    df.index,
    df["Children in HHS Care"],
)

plt.title("Children in HHS Care Over Time")
plt.xlabel("Date")
plt.ylabel("Children")
plt.grid(True)

plt.show()

# ----------------------------
# Correlation Heatmap
# ----------------------------

plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="Blues"
)

plt.title("Correlation Heatmap")

plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")