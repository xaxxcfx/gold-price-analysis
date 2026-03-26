#!/usr/bin/env python3
"""
Gold Price Analysis Script
Run this to generate all analysis and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add src to path
sys.path.append('src')

from data_processing import load_data, add_features, get_summary_stats
from visualization import plot_price_trend, plot_yearly_comparison
from model import prepare_features, train_model, save_model

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 60)
print("GOLD PRICE ANALYSIS")
print("=" * 60)

# 1. Load Data
print("\n[1/6] Loading data...")
df = load_data('data/GoldUSD.csv')
print(f"Loaded {len(df):,} records")

# 2. Quick Overview
print("\n[2/6] Data Overview:")
print(f"   Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
print(f"   Columns: {', '.join(df.columns.tolist())}")
print(f"\nFirst 5 rows:")
print(df.head())

# 3. Summary Statistics
print("\n[3/6] Summary Statistics:")
print(df[['Open', 'High', 'Low', 'Close', 'Volume']].describe().round(2))

# 4. Add Features
print("\n[4/6] Adding features...")
df = add_features(df)
print("Features added: Year, Month, Day, Quarter, Price_Change, Price_MA_7, Price_MA_30")

# 5. Generate Visualizations
print("\n[5/6] Generating visualizations...")

# Price trend
fig = plot_price_trend(df, save_path='results/price_trend.png')
plt.close()
print("Saved: results/price_trend.png")

# Yearly comparison
fig = plot_yearly_comparison(df, save_path='results/yearly_comparison.png')
plt.close()
print("Saved: results/yearly_comparison.png")

# Price distribution
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['Close'], kde=True, color='gold', ax=ax)
ax.set_title('Gold Price Distribution', fontsize=16, fontweight='bold')
ax.set_xlabel('Price (USD)')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('results/price_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: results/price_distribution.png")

# Monthly average prices
monthly_avg = df.groupby('Month')['Close'].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(monthly_avg['Month'], monthly_avg['Close'], marker='o', linewidth=2, color='gold')
ax.set_title('Average Gold Price by Month', fontsize=16, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Average Price (USD)')
ax.set_xticks(range(1, 13))
plt.tight_layout()
plt.savefig('results/monthly_average.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: results/monthly_average.png")

# 6. Train Model
print("\n[6/6] Training prediction model...")
try:
    X, y = prepare_features(df, target_col='Close', feature_cols=['Open', 'High', 'Low', 'Volume', 'Year', 'Month'])
    model, metrics, (X_test, y_test, y_pred) = train_model(X, y, model_type='random_forest')

    print(f"\nModel Performance:")
    print(f"   RMSE: ${metrics['rmse']:.2f}")
    print(f"   MAE: ${metrics['mae']:.2f}")
    print(f"   R2 Score: {metrics['r2']:.4f}")

    # Save model
    save_model(model, 'models/gold_price_model.pkl')
    print("Model saved to: models/gold_price_model.pkl")

    # Plot predictions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, color='gold')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    ax.set_title('Model Predictions vs Actual', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/model_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: results/model_predictions.png")

except Exception as e:
    print(f"Model training skipped: {e}")

# 7. Key Insights
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

min_price = df['Close'].min()
max_price = df['Close'].max()
avg_price = df['Close'].mean()

print(f"\nPrice Statistics:")
print(f"   Lowest Price: ${min_price:.2f} (on {df.loc[df['Close'].idxmin(), 'Date'].strftime('%Y-%m-%d')})")
print(f"   Highest Price: ${max_price:.2f} (on {df.loc[df['Close'].idxmax(), 'Date'].strftime('%Y-%m-%d')})")
print(f"   Average Price: ${avg_price:.2f}")
print(f"   Total Return: {((max_price - min_price) / min_price * 100):.1f}%")

# Best and worst years
yearly_returns = df.groupby('Year')['Close'].agg(['first', 'last'])
yearly_returns['return'] = (yearly_returns['last'] - yearly_returns['first']) / yearly_returns['first'] * 100
best_year = yearly_returns['return'].idxmax()
worst_year = yearly_returns['return'].idxmin()

print(f"\nBest/Worst Years:")
print(f"   Best Year: {best_year} (+{yearly_returns.loc[best_year, 'return']:.1f}%)")
print(f"   Worst Year: {worst_year} ({yearly_returns.loc[worst_year, 'return']:.1f}%)")

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
print("\nCheck the 'results/' folder for all generated charts.")
