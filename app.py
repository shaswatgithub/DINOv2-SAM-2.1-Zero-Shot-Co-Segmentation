import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings("ignore")

# ---------------- UI ---------------- #
st.set_page_config(page_title="ARIMA Dashboard", layout="wide")
st.title("🚀 Time Series Forecasting Dashboard (ARIMA + Auto ARIMA)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# ---------------- MAIN ---------------- #
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    # -------- AUTO COLUMN DETECTION -------- #
    # First column = date
    date_col = df.columns[0]

    # First numeric column = value
    numeric_cols = df.select_dtypes(include=np.number).columns
    value_col = numeric_cols[0]

    st.success(f"Using Date Column: {date_col}")
    st.success(f"Using Value Column: {value_col}")

    # -------- PREPROCESSING -------- #
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])

    df = df.set_index(date_col)
    df = df[[value_col]].astype(float).dropna()

    # Fix frequency
    try:
        df = df.asfreq(pd.infer_freq(df.index))
    except:
        df = df.resample('H').mean()

    st.subheader("📈 Time Series")
    st.line_chart(df[value_col])

    # -------- STATIONARITY -------- #
    st.subheader("📊 ADF Test")
    result = adfuller(df[value_col])

    st.write(f"ADF Statistic: {result[0]:.4f}")
    st.write(f"p-value: {result[1]:.4f}")

    if result[1] < 0.05:
        st.success("Data is stationary")
    else:
        st.warning("Data is NOT stationary")

    # -------- TRAIN TEST SPLIT -------- #
    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]

    st.write(f"Train: {len(train)} | Test: {len(test)}")

    # -------- TABS -------- #
    tab1, tab2, tab3 = st.tabs(["Manual ARIMA", "Auto ARIMA", "Diagnostics"])

    # =========================
    # 🔹 MANUAL ARIMA
    # =========================
    with tab1:

        st.subheader("Manual ARIMA")

        p = st.slider("p", 0, 5, 2)
        d = st.slider("d", 0, 2, 1)
        q = st.slider("q", 0, 5, 2)

        if st.button("Run Manual ARIMA"):

            model = ARIMA(train[value_col], order=(p, d, q))
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=len(test))

            rmse = np.sqrt(mean_squared_error(test[value_col], forecast))
            mae = mean_absolute_error(test[value_col], forecast)

            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"MAE: {mae:.2f}")

            # Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(train.index, train[value_col], label="Train")
            ax.plot(test.index, test[value_col], label="Test")
            ax.plot(test.index, forecast, label="Forecast", color="red")
            ax.legend()
            st.pyplot(fig)

            # Residuals
            st.subheader("Residuals")
            residuals = model_fit.resid

            fig, ax = plt.subplots()
            ax.plot(residuals)
            st.pyplot(fig)

            fig, ax = plt.subplots()
            plot_acf(residuals, ax=ax)
            st.pyplot(fig)

    # =========================
    # 🔹 AUTO ARIMA
    # =========================
    with tab2:

        st.subheader("Auto ARIMA")

        seasonal = st.checkbox("Seasonal", value=True)
        m = st.number_input("Seasonal Period (m)", value=24)

        if st.button("Run Auto ARIMA"):

            with st.spinner("Finding best model..."):

                model = pm.auto_arima(
                    train[value_col],
                    seasonal=seasonal,
                    m=m,
                    stepwise=True,
                    suppress_warnings=True
                )

                st.write("Best Order:", model.order)
                if seasonal:
                    st.write("Seasonal Order:", model.seasonal_order)

                forecast = model.predict(n_periods=len(test))

                rmse = np.sqrt(mean_squared_error(test[value_col], forecast))
                mae = mean_absolute_error(test[value_col], forecast)

                st.write(f"RMSE: {rmse:.2f}")
                st.write(f"MAE: {mae:.2f}")

                # Plot
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(train.index, train[value_col], label="Train")
                ax.plot(test.index, test[value_col], label="Test")
                ax.plot(test.index, forecast, label="Forecast", color="orange")
                ax.legend()
                st.pyplot(fig)

    # =========================
    # 🔹 DIAGNOSTICS
    # =========================
    with tab3:

        st.subheader("ACF & PACF")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            plot_acf(df[value_col], lags=40, ax=ax)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            plot_pacf(df[value_col], lags=40, ax=ax)
            st.pyplot(fig)

else:
    st.info("Upload your dataset to begin")

st.caption("ARIMA Dashboard • Fully Robust Version")