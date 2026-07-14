import pandas as pd
import matplotlib.pyplot as plt
import math
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================
# Load Dataset
# ============================

df = pd.read_csv("data/feature_data.csv")

print("Dataset Loaded Successfully!\n")
print(df.head())

# ============================
# Target Column
# ============================

target = "Children in HHS Care"

# Features and Target
X = df.drop(columns=["Date", target])
y = df[target]

# ============================
# Time Series Train-Test Split
# ============================

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ============================
# Train Random Forest
# ============================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

# ============================
# Prediction
# ============================

pred = model.predict(X_test)

# ============================
# Evaluation
# ============================

mae = mean_absolute_error(y_test, pred)

mse = mean_squared_error(y_test, pred)
rmse = math.sqrt(mse)

r2 = r2_score(y_test, pred)

print("\n==============================")
print("Random Forest Results")
print("==============================")

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

# ============================
# Save Model
# ============================

joblib.dump(model, "models/random_forest.pkl")

print("\nModel Saved Successfully!")

# ============================
# Actual vs Prediction Plot
# ============================

plt.figure(figsize=(15,6))

plt.plot(
    y_test.values,
    label="Actual",
    linewidth=2
)

plt.plot(
    pred,
    label="Predicted",
    linewidth=2
)

plt.title("Random Forest Prediction")
plt.xlabel("Days")
plt.ylabel("Children in HHS Care")
plt.legend()
plt.grid(True)

plt.show()

# ============================
# Feature Importance
# ============================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance\n")
print(importance)

plt.figure(figsize=(10,6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Feature Importance")
plt.xlabel("Importance")
plt.gca().invert_yaxis()

plt.show()

print("\nRandom Forest Training Completed Successfully!")