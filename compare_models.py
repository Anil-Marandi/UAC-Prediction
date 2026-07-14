import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Results of all models
# ----------------------------

results = pd.DataFrame({

    "Model":[
        "Random Forest",
        "Gradient Boosting",
        "ARIMA"
    ],

    "MAE":[
        67.14,
        65.60,
        214.58
    ],

    "RMSE":[
        88.10,
        85.32,
        265.12
    ],

    "R2":[
        0.7497,
        0.7653,
        -1.2739
    ]

})

print(results)

results.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

# ----------------------------
# MAE
# ----------------------------

plt.figure(figsize=(8,5))

plt.bar(results["Model"],results["MAE"])

plt.title("MAE Comparison")

plt.ylabel("MAE")

plt.grid(axis="y")

plt.savefig("outputs/mae_comparison.png")

plt.show()

# ----------------------------
# RMSE
# ----------------------------

plt.figure(figsize=(8,5))

plt.bar(results["Model"],results["RMSE"])

plt.title("RMSE Comparison")

plt.ylabel("RMSE")

plt.grid(axis="y")

plt.savefig("outputs/rmse_comparison.png")

plt.show()

# ----------------------------
# R2
# ----------------------------

plt.figure(figsize=(8,5))

plt.bar(results["Model"],results["R2"])

plt.title("R² Comparison")

plt.ylabel("R²")

plt.grid(axis="y")

plt.savefig("outputs/r2_comparison.png")

plt.show()

print("\nComparison Completed Successfully!")