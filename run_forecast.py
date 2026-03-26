#!/usr/bin/env python3
"""
Gold Price Forecasting Script
Generates forecasts using multiple models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append('src')

from data_processing import load_data, add_features
from forecasting import (
    exponential_smoothing_forecast,
    arima_forecast,
    moving_average_forecast,
    create_forecast_plot,
    evaluate_forecast
)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 70)
print("GOLD PRICE FORECASTING")
print("=" * 70)

# 1. Load Data
print("\n[1/4] Loading data...")
df = load_data('data/GoldUSD.csv')
df = df.sort_values('Date')
print(f"Loaded {len(df):,} records from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")

# 2. Prepare forecast parameters
FORECAST_DAYS = 90  # Forecast next 90 days

print(f"\n[2/4] Generating {FORECAST_DAYS}-day forecasts...")

# Split data for backtesting
# Use last 90 days as test set
train_df = df[:-90].copy()
test_df = df[-90:].copy()

print(f"Training on: {len(train_df)} days")
print(f"Testing on: {len(test_df)} days")

# 3. Train forecasts on training data
print("\n[3/4] Training forecast models...")

# Exponential Smoothing
print("  - Training Exponential Smoothing...")
try:
    es_forecast, es_conf = exponential_smoothing_forecast(train_df, periods=FORECAST_DAYS)
    es_metrics = evaluate_forecast(test_df['Close'].values[:FORECAST_DAYS], es_forecast)
    print(f"    RMSE: ${es_metrics['rmse']:.2f}, MAPE: {es_metrics['mape']:.2f}%")
    es_success = True
except Exception as e:
    print(f"    Failed: {e}")
    es_success = False

# ARIMA
print("  - Training ARIMA...")
try:
    arima_forecast_vals, arima_conf = arima_forecast(train_df, periods=FORECAST_DAYS)
    arima_metrics = evaluate_forecast(test_df['Close'].values[:FORECAST_DAYS], arima_forecast_vals)
    print(f"    RMSE: ${arima_metrics['rmse']:.2f}, MAPE: {arima_metrics['mape']:.2f}%")
    arima_success = True
except Exception as e:
    print(f"    Failed: {e}")
    arima_success = False

# Moving Average
print("  - Training Moving Average...")
try:
    ma_forecast = moving_average_forecast(train_df, window=30, periods=FORECAST_DAYS)
    ma_metrics = evaluate_forecast(test_df['Close'].values[:FORECAST_DAYS], ma_forecast)
    print(f"    RMSE: ${ma_metrics['rmse']:.2f}, MAPE: {ma_metrics['mape']:.2f}%")
    ma_success = True
except Exception as e:
    print(f"    Failed: {e}")
    ma_success = False

# 4. Create visualizations
print("\n[4/4] Creating visualizations...")

# Prepare forecast results dict
forecast_results = {}
if es_success:
    forecast_results['Exponential Smoothing'] = {
        'forecast': es_forecast,
        'confidence': es_conf,
        'metrics': es_metrics
    }
if arima_success:
    forecast_results['ARIMA'] = {
        'forecast': arima_forecast_vals,
        'confidence': arima_conf,
        'metrics': arima_metrics
    }
if ma_success:
    forecast_results['Moving Average'] = {
        'forecast': ma_forecast,
        'metrics': ma_metrics
    }

# Create forecast comparison plot
fig = create_forecast_plot(df, forecast_results, periods=FORECAST_DAYS,
                            save_path='results/forecast_comparison.png')
plt.close()
print("Saved: results/forecast_comparison.png")

# Create backtest plot
fig, ax = plt.subplots(figsize=(14, 7))

# Plot actual test data
ax.plot(test_df['Date'], test_df['Close'], label='Actual', color='blue', linewidth=3)

# Plot forecasts
last_train_date = train_df['Date'].max()
future_dates = pd.date_range(start=last_train_date + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq='D')

colors = ['orange', 'green', 'red']
for i, (model_name, forecast_data) in enumerate(forecast_results.items()):
    forecast = forecast_data['forecast']
    ax.plot(future_dates, forecast, label=f'{model_name} Forecast',
            color=colors[i % len(colors)], linestyle='--', linewidth=2)

ax.set_title('90-Day Forecast Backtest: Model Comparison vs Actual', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Price (USD)', fontsize=12)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/forecast_backtest.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: results/forecast_backtest.png")

# 5. Generate final forecast (using ALL data)
print("\n" + "=" * 70)
print("FINAL FORECAST (Next 90 Days)")
print("=" * 70)

print("\nGenerating final forecast using all available data...")

final_forecasts = {}

# Exponential Smoothing on full data
try:
    es_final, es_final_conf = exponential_smoothing_forecast(df, periods=FORECAST_DAYS)
    final_forecasts['Exponential Smoothing'] = {
        'forecast': es_final,
        'confidence': es_final_conf
    }
except Exception as e:
    print(f"ES failed: {e}")

# ARIMA on full data
try:
    arima_final, arima_final_conf = arima_forecast(df, periods=FORECAST_DAYS)
    final_forecasts['ARIMA'] = {
        'forecast': arima_final,
        'confidence': arima_final_conf
    }
except Exception as e:
    print(f"ARIMA failed: {e}")

# Print predictions
last_date = df['Date'].max()
last_price = df['Close'].iloc[-1]

print(f"\nLast known price: ${last_price:.2f} (on {last_date.strftime('%Y-%m-%d')})")
print("\nForecasted prices:")
print("-" * 70)
print(f"{'Date':<15} {'Exp Smoothing':<20} {'ARIMA':<20}")
print("-" * 70)

# Calculate future dates
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq='D')

# Print key dates (every 30 days)
for i in [29, 59, 89]:  # Day 30, 60, 90
    date_str = future_dates[i].strftime('%Y-%m-%d')
    es_price = final_forecasts.get('Exponential Smoothing', {}).get('forecast', [np.nan] * FORECAST_DAYS)[i]
    arima_price = final_forecasts.get('ARIMA', {}).get('forecast', [np.nan] * FORECAST_DAYS)[i]

    es_str = f"${es_price:.2f}" if not np.isnan(es_price) else "N/A"
    arima_str = f"${arima_price:.2f}" if not np.isnan(arima_price) else "N/A"

    print(f"{date_str:<15} {es_str:<20} {arima_str:<20}")

# Calculate predicted returns
try:
    es_30d = final_forecasts['Exponential Smoothing']['forecast'][29]
    es_return = ((es_30d - last_price) / last_price) * 100
    print(f"\nExpected 30-day return (Exponential Smoothing): {es_return:+.1f}%")
except:
    pass

try:
    arima_30d = final_forecasts['ARIMA']['forecast'][29]
    arima_return = ((arima_30d - last_price) / last_price) * 100
    print(f"Expected 30-day return (ARIMA): {arima_return:+.1f}%")
except:
    pass

# Model Performance Summary
print("\n" + "=" * 70)
print("MODEL PERFORMANCE (Backtest)")
print("=" * 70)

if es_success:
    print(f"\nExponential Smoothing:")
    print(f"  RMSE: ${es_metrics['rmse']:.2f}")
    print(f"  MAE:  ${es_metrics['mae']:.2f}")
    print(f"  MAPE: {es_metrics['mape']:.2f}%")

if arima_success:
    print(f"\nARIMA:")
    print(f"  RMSE: ${arima_metrics['rmse']:.2f}")
    print(f"  MAE:  ${arima_metrics['mae']:.2f}")
    print(f"  MAPE: {arima_metrics['mape']:.2f}%")

if ma_success:
    print(f"\nMoving Average:")
    print(f"  RMSE: ${ma_metrics['rmse']:.2f}")
    print(f"  MAE:  ${ma_metrics['mae']:.2f}")
    print(f"  MAPE: {ma_metrics['mape']:.2f}%")

print("\n" + "=" * 70)
print("Forecasting Complete!")
print("=" * 70)
print("\nCheck the 'results/' folder for forecast visualizations.")
