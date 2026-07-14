import pandas as pd
import matplotlib.pyplot as plt
import math
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================
# Load Dataset
# ============================

df = pd.read_csv("data/feature_data.csv")

print("Dataset Loaded Successfully!\n")

target = "Children in HHS Care"

X = df.drop(columns=["Date", target])

y = df[target]

# ============================
# Train Test Split
# ============================

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# ============================
# Gradient Boosting Model
# ============================

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
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
print("Gradient Boosting Results")
print("==============================")

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

# ============================
# Save Model
# ============================

joblib.dump(model, "models/gradient_boosting.pkl")

print("\nGradient Boosting Model Saved!")

# ============================
# Actual vs Prediction
# ============================

plt.figure(figsize=(15,6))

plt.plot(y_test.values, label="Actual")

plt.plot(pred, label="Predicted")

plt.title("Gradient Boosting Prediction")

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

plt.gca().invert_yaxis()

plt.title("Gradient Boosting Feature Importance")

plt.show()

print("\nGradient Boosting Training Completed Successfully!")