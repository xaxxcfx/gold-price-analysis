# Gold Price Dynamics Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive data science project analyzing gold price trends from 2000-2026 with machine learning predictions and time series forecasting.

## Key Highlights

- **6,399 records** spanning 25+ years of daily gold prices
- **ML Model** with R² = 0.9999 accuracy for price prediction
- **3 Forecasting Models** (Exponential Smoothing, ARIMA, Moving Average)
- **Interactive Dashboard** built with Streamlit
- **+2,007.9%** total return from 2000 to 2026

## Project Structure

```
gold-price-analysis/
├── data/                   # Dataset folder
├── notebooks/              # Jupyter notebooks
│   └── 01_exploratory_analysis.ipynb
├── src/                    # Source code modules
│   ├── data_processing.py  # Data loading and cleaning
│   ├── visualization.py    # Chart generation
│   ├── model.py           # ML model training
│   └── forecasting.py      # Time series forecasting
├── models/                 # Saved ML models
├── results/                # Generated charts
├── run_analysis.py         # Main analysis script
├── run_forecast.py         # Forecasting script
├── dashboard.py            # Streamlit web app
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── PROJECT_SUMMARY.md     # Detailed project report
```

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/gold-price-analysis.git
cd gold-price-analysis

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Analysis

```bash
# Complete analysis with ML model
python run_analysis.py

# Time series forecasting (90-day predictions)
python run_forecast.py

# Launch interactive dashboard
streamlit run dashboard.py
```

## Features

### 1. Exploratory Data Analysis
- Price trend visualization
- Seasonal pattern analysis
- Statistical summaries
- Year-over-year comparisons

### 2. Machine Learning
- **Algorithm**: Random Forest Regressor
- **Features**: Open, High, Low, Volume, Year, Month
- **Performance**:
  - RMSE: $8.78
  - MAE: $4.46
  - R² Score: 0.9999

### 3. Time Series Forecasting

| Model | RMSE | MAPE | Use Case |
|-------|------|------|----------|
| Exponential Smoothing | $367.68 | 5.44% | Short-term forecasts |
| ARIMA | $579.67 | 8.94% | Long-term trends |
| Moving Average | $745.45 | 12.99% | Simple smoothing |

### 4. Interactive Dashboard
- Real-time data exploration
- Adjustable date ranges
- Multiple visualization types
- Forecast comparisons

## Sample Results

### Key Insights

**Price Statistics:**
- Lowest Price: $255.10 (Feb 15, 2001)
- Highest Price: $5,377.30 (Mar 3, 2026)
- Average Price: $1,274.96
- Total Return: +2,007.9%

**Best/Worst Years:**
- Best Year: 2025 (+62.7%)
- Worst Year: 2013 (-28.8%)

**90-Day Forecast** (March 2026):
- Exponential Smoothing: $5,622 (+4.6% in 30 days)
- ARIMA: $5,362 (-0.3% in 30 days)

## Generated Charts

All visualizations are saved in the `results/` folder:

1. `price_trend.png` - Historical price trajectory
2. `yearly_comparison.png` - Annual distributions
3. `price_distribution.png` - Statistical distribution
4. `monthly_average.png` - Seasonal patterns
5. `model_predictions.png` - ML accuracy
6. `forecast_comparison.png` - Forecast comparison
7. `forecast_backtest.png` - Model validation

## Tech Stack

- **Python 3.10+**
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Machine learning
- **Statsmodels** - Time series analysis (ARIMA)
- **Matplotlib & Seaborn** - Visualization
- **Streamlit** - Interactive dashboard
- **Jupyter** - Notebooks for exploration

## Dataset

The dataset includes daily gold prices with:
- **Date**: Trading date
- **Open**: Opening price
- **High**: Highest price of the day
- **Low**: Lowest price of the day
- **Close**: Closing price
- **Volume**: Trading volume

## Model Training

The Random Forest model uses:
- Input features: Open, High, Low, Volume, Year, Month
- Target: Close price
- Train/Test split: 80/20
- Model saved to: `models/gold_price_model.pkl`

## Forecasting

Time series models predict future prices using:
- Historical patterns
- Seasonal trends
- Moving averages
- Auto-regressive components

## Future Enhancements

- [ ] Add macroeconomic indicators (inflation, interest rates, USD index)
- [ ] Implement LSTM neural networks
- [ ] Create automated trading signals
- [ ] Add sentiment analysis from news
- [ ] Deploy dashboard to cloud

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Author

Your Name - [your.email@example.com](mailto:your.email@example.com)

---

**⭐ Star this repo if you find it useful!**
