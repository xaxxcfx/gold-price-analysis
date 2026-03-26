#!/usr/bin/env python3
"""
Interactive Streamlit Dashboard for Gold Price Analysis
Run with: streamlit run dashboard.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import streamlit as st

sys.path.append('src')

from data_processing import load_data, add_features, get_summary_stats
from visualization import create_interactive_chart
from forecasting import exponential_smoothing_forecast, arima_forecast

# Page config
st.set_page_config(
    page_title="Gold Price Analysis Dashboard",
    page_icon=":moneybag:",
    layout="wide"
)

# Title
st.title("Gold Price Dynamics Analysis")
st.markdown("Analysis of gold prices from 2000 to 2026 with ML predictions")

# Load data
@st.cache_data
def load_cached_data():
    df = load_data('data/GoldUSD.csv')
    df = add_features(df)
    return df

df = load_cached_data()

# Sidebar
st.sidebar.header("Dashboard Controls")
selected_view = st.sidebar.radio(
    "Select View",
    ["Overview", "Time Series Analysis", "Forecasting", "Model Performance"]
)

if selected_view == "Overview":
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Date Range", f"{df['Date'].dt.year.min()}-{df['Date'].dt.year.max()}")
    with col3:
        st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
    with col4:
        total_return = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
        st.metric("Total Return", f"{total_return:.1f}%")

    st.subheader("Price Summary")
    st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].describe().round(2))

    st.subheader("Recent Data")
    st.dataframe(df.tail(10)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].reset_index(drop=True))

elif selected_view == "Time Series Analysis":
    st.header("Price Trends")

    # Date range selector
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

    # Filter data
    mask = (df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))
    filtered_df = df.loc[mask]

    # Chart type
    chart_type = st.selectbox("Chart Type", ["Line Chart", "Candlestick", "Moving Averages"])

    if chart_type == "Line Chart":
        st.line_chart(filtered_df.set_index('Date')['Close'])

    elif chart_type == "Moving Averages":
        ma_days = st.slider("Moving Average Window (days)", 5, 100, 30)
        filtered_df[f'MA_{ma_days}'] = filtered_df['Close'].rolling(window=ma_days).mean()

        chart_data = filtered_df.set_index('Date')[['Close', f'MA_{ma_days}']]
        st.line_chart(chart_data)

    # Yearly comparison
    st.subheader("Yearly Performance")
    yearly = df.groupby('Year')['Close'].agg(['first', 'last', 'min', 'max'])
    yearly['Return'] = ((yearly['last'] - yearly['first']) / yearly['first'] * 100).round(1)
    st.dataframe(yearly[['min', 'max', 'Return']])

elif selected_view == "Forecasting":
    st.header("Price Forecasting")

    forecast_days = st.slider("Forecast Period (days)", 7, 180, 30)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Exponential Smoothing")
        try:
            forecast, conf = exponential_smoothing_forecast(df, periods=forecast_days)
            st.metric("Forecasted Price (End)", f"${forecast[-1]:.2f}")

            # Create future dates
            last_date = df['Date'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')

            # Plot
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': forecast
            })

            chart_data = pd.concat([
                df[['Date', 'Close']].rename(columns={'Close': 'Price'}),
                forecast_df.rename(columns={'Forecast': 'Price'})
            ])
            st.line_chart(chart_data.set_index('Date'))

        except Exception as e:
            st.error(f"Forecast error: {e}")

    with col2:
        st.subheader("ARIMA")
        try:
            forecast, conf = arima_forecast(df, periods=forecast_days)
            st.metric("Forecasted Price (End)", f"${forecast[-1]:.2f}")

            last_date = df['Date'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')

            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': forecast
            })

            chart_data = pd.concat([
                df[['Date', 'Close']].rename(columns={'Close': 'Price'}),
                forecast_df.rename(columns={'Forecast': 'Price'})
            ])
            st.line_chart(chart_data.set_index('Date'))

        except Exception as e:
            st.error(f"Forecast error: {e}")

elif selected_view == "Model Performance":
    st.header("Machine Learning Model Performance")

    # Load model results (these would be saved from training)
    st.info("Model training metrics from previous run:")

    metrics_data = {
        'Model': ['Random Forest', 'Exponential Smoothing', 'ARIMA', 'Moving Average'],
        'RMSE': ['~$8.78', '$367.68', '$579.67', '$745.45'],
        'MAPE': ['~0.16%', '5.44%', '8.94%', '12.99%']
    }

    st.dataframe(metrics_data)

    st.subheader("Key Insights")
    st.markdown("""
    - **Random Forest** achieved near-perfect accuracy (R² = 0.9999)
    - **Exponential Smoothing** best for time series forecasting (MAPE: 5.44%)
    - **ARIMA** provides stable long-term forecasts
    - **Moving Average** simplest but highest error
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit")

if __name__ == "__main__":
    pass
