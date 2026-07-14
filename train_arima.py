import pandas as pd
import matplotlib.pyplot as plt
import math
import joblib

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ====================================

print("Loading Dataset...")

df = pd.read_csv("data/clean_data.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

target = "Children in HHS Care"

series = df[target]

# ====================================

split = int(len(series)*0.8)

train = series[:split]

test = series[split:]

print("Training :",len(train))

print("Testing :",len(test))

# ====================================

print("Training ARIMA Model...")

model = ARIMA(train,
              order=(7,1,1))

model_fit = model.fit()

# ====================================

forecast = model_fit.forecast(
    steps=len(test)
)

# ====================================

mae = mean_absolute_error(
    test,
    forecast
)

mse = mean_squared_error(
    test,
    forecast
)

rmse = math.sqrt(mse)

r2 = r2_score(
    test,
    forecast
)

print("\n=========== ARIMA RESULTS ==========")

print(f"MAE : {mae:.2f}")

print(f"MSE : {mse:.2f}")

print(f"RMSE : {rmse:.2f}")

print(f"R2 : {r2:.4f}")

# ====================================

joblib.dump(
    model_fit,
    "models/arima.pkl"
)

print("\nARIMA Model Saved!")

# ====================================

plt.figure(figsize=(15,6))

plt.plot(
    test.values,
    label="Actual"
)

plt.plot(
    forecast.values,
    label="Forecast"
)

plt.legend()

plt.title("ARIMA Forecast")

plt.grid(True)

plt.show()

print("\nARIMA Completed Successfully!")