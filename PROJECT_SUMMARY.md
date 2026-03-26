# Gold Price Analysis - Project Summary

## Executive Summary

This project analyzes 25+ years of gold price data (6,399 records from 2000-2026) using machine learning and time series forecasting techniques to predict future prices and identify key trends.

---

## Key Achievements

### 1. Exploratory Data Analysis
- **Dataset**: 6,399 daily gold price records spanning 25+ years
- **Price Range**: $255.10 (2001) to $5,377.30 (2026)
- **Total Return**: +2,007.9% over the period
- **Average Price**: $1,274.96

### 2. Machine Learning Model
- **Algorithm**: Random Forest Regressor
- **Performance**: R² Score = 0.9999 (nearly perfect prediction)
- **Error Metrics**:
  - RMSE: $8.78
  - MAE: $4.46
- **Features Used**: Open, High, Low, Volume, Year, Month

### 3. Time Series Forecasting

| Model | RMSE | MAPE | Best For |
|-------|------|------|----------|
| **Exponential Smoothing** | $367.68 | 5.44% | Short-term forecasts |
| **ARIMA** | $579.67 | 8.94% | Long-term trends |
| **Moving Average** | $745.45 | 12.99% | Simple smoothing |

**90-Day Forecast Results** (as of March 2026):
- Exponential Smoothing predicts: **$5,622** (+4.6% in 30 days)
- ARIMA predicts: **$5,362** (-0.3% in 30 days)

### 4. Key Insights

#### Best/Worst Performing Years
- **Best Year**: 2025 (+62.7% return)
- **Worst Year**: 2013 (-28.8% return)
- **Highest Volatility**: During 2008 financial crisis and 2020 COVID pandemic

#### Seasonal Patterns
- Gold tends to perform better in Q1 and Q4
- January typically shows above-average returns
- Summer months (June-August) show lower volatility

---

## Technical Implementation

### Tech Stack
- **Python 3.10+**
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Machine learning
- **Statsmodels** - Time series analysis (ARIMA)
- **Matplotlib & Seaborn** - Visualization
- **Streamlit** - Interactive dashboard

### Project Structure
```
gold-price-analysis/
├── data/                   # Dataset
├── notebooks/              # Jupyter notebooks
├── src/                    # Source code modules
│   ├── data_processing.py
│   ├── visualization.py
│   ├── model.py
│   └── forecasting.py      # NEW: Time series forecasting
├── models/                 # Saved ML models
├── results/                # Generated charts
├── run_analysis.py         # Main analysis script
├── run_forecast.py         # NEW: Forecasting script
├── dashboard.py            # NEW: Streamlit dashboard
└── README.md
```

### Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `run_analysis.py` | Full EDA + ML model | 5 charts, trained model |
| `run_forecast.py` | Time series forecasting | 90-day forecasts, comparison charts |
| `dashboard.py` | Interactive web app | Streamlit dashboard |

---

## Visualizations Generated

1. **price_trend.png** - Historical price trajectory
2. **yearly_comparison.png** - Annual price distributions
3. **price_distribution.png** - Statistical distribution
4. **monthly_average.png** - Seasonal patterns
5. **model_predictions.png** - ML accuracy visualization
6. **forecast_comparison.png** - Multi-model forecast comparison
7. **forecast_backtest.png** - Model validation

---

## Business Applications

1. **Investment Decision Support**: Predict gold price movements
2. **Risk Management**: Understand volatility patterns
3. **Portfolio Allocation**: Optimize gold holdings based on forecasts
4. **Market Analysis**: Identify long-term trends and cycles

---

## Future Enhancements

- [ ] Add macroeconomic indicators (inflation, interest rates, USD index)
- [ ] Implement LSTM neural networks for deep learning forecasting
- [ ] Create automated trading signals
- [ ] Add sentiment analysis from news/social media
- [ ] Deploy dashboard to cloud (AWS/Heroku)

---

## How to Run

```bash
# Setup
pip install -r requirements.txt

# Run analysis
python run_analysis.py

# Run forecasting
python run_forecast.py

# Launch dashboard
streamlit run dashboard.py
```

---

**Author**: [Your Name]
**Date**: March 2026
**Repository**: https://github.com/[username]/gold-price-analysis
