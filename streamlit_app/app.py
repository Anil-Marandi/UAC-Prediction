import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Unaccompanied Alien Children Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("📊 Unaccompanied Alien Children Prediction Dashboard")

st.markdown("---")

# -----------------------------
# Sidebar
# -----------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset",
        "Statistics",
        "Visualizations",
        "Model Comparison",
        "Prediction"
    ]
)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/feature_data.csv")

# -----------------------------
# Load Models
# -----------------------------
rf_model = joblib.load("models/random_forest.pkl")
gb_model = joblib.load("models/gradient_boosting.pkl")
arima_model = joblib.load("models/arima.pkl")



# ===================================================
# HOME
# ===================================================

if page == "Home":

    st.header("Welcome")

    st.write("""
This dashboard predicts **Children in HHS Care**
using multiple Machine Learning models.

### Available Models

- Random Forest
- Gradient Boosting
- ARIMA

### Features

- Dataset Overview
- Statistics
- Visualizations
- Model Comparison
- Prediction
    """)

    st.success("Project Developed using Python, Scikit-Learn and Streamlit.")


# ===================================================
# DATASET
# ===================================================

elif page == "Dataset":

    st.header("📂 Dataset Overview")

    st.write("Dataset Shape")

    st.write(df.shape)

    st.write("Columns")

    st.write(list(df.columns))

    st.write("First 10 Rows")

    st.dataframe(df.head(10))

    st.write("Last 10 Rows")

    st.dataframe(df.tail(10))


# ===================================================
# STATISTICS
# ===================================================

elif page == "Statistics":

    st.header("📈 Dataset Statistics")

    # Numeric columns only
    numeric_df = df.select_dtypes(include=["number"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("---")

    st.subheader("Statistical Summary")

    st.dataframe(numeric_df.describe().T)

    st.markdown("---")

    st.subheader("Mean Values")

    st.dataframe(numeric_df.mean().to_frame("Mean"))

    st.subheader("Median Values")

    st.dataframe(numeric_df.median().to_frame("Median"))

    st.subheader("Standard Deviation")

    st.dataframe(numeric_df.std().to_frame("Std"))


# ===================================================
# VISUALIZATIONS
# ===================================================

elif page == "Visualizations":

    st.header("📊 Data Visualizations")

    # -----------------------------
    # Time Series Plot
    # -----------------------------
    st.subheader("Children in HHS Care Over Time")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        pd.to_datetime(df["Date"]),
        df["Children in HHS Care"],
        color="blue"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Children in HHS Care")
    ax.grid(True)

    st.pyplot(fig)

    # -----------------------------
    # Correlation Heatmap
    # -----------------------------
    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(12,8))

    sns.heatmap(
        df.select_dtypes(include="number").corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    # -----------------------------
    # Histogram
    # -----------------------------
    st.subheader("Children in HHS Care Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.hist(
        df["Children in HHS Care"],
        bins=20
    )

    ax.set_xlabel("Children in HHS Care")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    # -----------------------------
    # Box Plot
    # -----------------------------
    st.subheader("Box Plot")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.boxplot(df["Children in HHS Care"])

    st.pyplot(fig)


# ===================================================
# MODEL COMPARISON
# ===================================================

elif page == "Model Comparison":

    st.header("🏆 Model Comparison")

    comparison = pd.DataFrame({
        "Model": [
            "Random Forest",
            "Gradient Boosting",
            "ARIMA"
        ],
        "MAE": [
            67.14,
            65.60,
            214.58
        ],
        "RMSE": [
            88.10,
            85.32,
            265.12
        ],
        "R2": [
            0.7497,
            0.7653,
            -1.2739
        ]
    })

    st.dataframe(comparison)

    st.markdown("---")

    st.subheader("R² Score Comparison")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(
        comparison["Model"],
        comparison["R2"]
    )

    ax.set_ylabel("R²")

    st.pyplot(fig)

    st.subheader("RMSE Comparison")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(
        comparison["Model"],
        comparison["RMSE"]
    )

    ax.set_ylabel("RMSE")

    st.pyplot(fig)

    st.subheader("MAE Comparison")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(
        comparison["Model"],
        comparison["MAE"]
    )

    ax.set_ylabel("MAE")

    st.pyplot(fig)

    best = comparison.sort_values(
        by="R2",
        ascending=False
    ).iloc[0]

    st.success(
        f"Best Model : {best['Model']} (R² = {best['R2']:.4f})"
    )


# ===================================================
# PREDICTION
# ===================================================

elif page == "Prediction":

    st.header("🔮 Prediction")

    model_name = st.selectbox(
        "Select Model",
        [
            "Random Forest",
            "Gradient Boosting"
        ]
    )

    lag1 = st.number_input("Lag_1", value=2400.0)
    lag7 = st.number_input("Lag_7", value=2400.0)
    lag14 = st.number_input("Lag_14", value=2400.0)

    rolling7 = st.number_input("Rolling Mean 7", value=2400.0)
    rolling14 = st.number_input("Rolling Mean 14", value=2400.0)
    rollingstd7 = st.number_input("Rolling Std 7", value=50.0)

    cbp = st.number_input("Children in CBP custody", value=200.0)
    transfer = st.number_input("Children transferred out of CBP custody", value=150.0)
    apprehended = st.number_input("Children apprehended and placed in CBP custody", value=100.0)
    discharged = st.number_input("Children discharged from HHS Care", value=100.0)

    netpressure = st.number_input("Net Pressure", value=50.0)

    day = st.number_input("Day", value=15)
    month = st.number_input("Month", value=7)
    weekday = st.number_input("Weekday", value=2)
    week = st.number_input("Week", value=28)

    if st.button("Predict"):

        X = pd.DataFrame([[
            apprehended,
            cbp,
            transfer,
            discharged,
            lag1,
            lag7,
            lag14,
            rolling7,
            rolling14,
            rollingstd7,
            netpressure,
            day,
            month,
            weekday,
            week
        ]],
        columns=[
            "Children apprehended and placed in CBP custody*",
            "Children in CBP custody",
            "Children transferred out of CBP custody",
            "Children discharged from HHS Care",
            "Lag_1",
            "Lag_7",
            "Lag_14",
            "RollingMean7",
            "RollingMean14",
            "RollingStd7",
            "NetPressure",
            "Day",
            "Month",
            "Weekday",
            "Week"
        ])

        if model_name == "Random Forest":
            prediction = rf_model.predict(X)[0]
        else:
            prediction = gb_model.predict(X)[0]

        st.success(
            f"Predicted Children in HHS Care : {prediction:.2f}"
        )