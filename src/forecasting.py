# Forecasting Module

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def create_time_series_features(df):
    """
    Create features for time series forecasting.

    Args:
        df: DataFrame with Date and Close columns

    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    df = df.sort_values('Date')

    # Lag features
    for lag in [1, 3, 7, 30]:
        df[f'lag_{lag}'] = df['Close'].shift(lag)

    # Rolling statistics
    df['rolling_mean_7'] = df['Close'].rolling(window=7).mean()
    df['rolling_mean_30'] = df['Close'].rolling(window=30).mean()
    df['rolling_std_30'] = df['Close'].rolling(window=30).std()

    # Time-based features
    df['days_from_start'] = (df['Date'] - df['Date'].min()).dt.days

    return df

def exponential_smoothing_forecast(df, periods=30):
    """
    Forecast using Triple Exponential Smoothing (Holt-Winters).

    Args:
        df: DataFrame with Close prices
        periods: Number of days to forecast

    Returns:
        forecast_values, confidence_intervals
    """
    # Prepare data
    prices = df['Close'].values

    # Fit model
    model = ExponentialSmoothing(
        prices,
        trend='add',
        seasonal='add',
        seasonal_periods=252  # Trading days in a year
    )
    fitted = model.fit()

    # Forecast
    forecast = fitted.forecast(periods)

    # Calculate confidence intervals (simplified)
    residuals = fitted.resid
    std_residuals = np.std(residuals)
    conf_int = np.column_stack([
        forecast - 1.96 * std_residuals,
        forecast + 1.96 * std_residuals
    ])

    return forecast, conf_int

def arima_forecast(df, periods=30):
    """
    Forecast using ARIMA model.

    Args:
        df: DataFrame with Close prices
        periods: Number of days to forecast

    Returns:
        forecast_values, confidence_intervals
    """
    # Prepare data
    prices = df['Close'].values

    # Fit ARIMA model (5,1,0) - auto-regressive with differencing
    model = ARIMA(prices, order=(5, 1, 0))
    fitted = model.fit()

    # Forecast
    forecast = fitted.forecast(steps=periods)

    # Get confidence intervals
    forecast_result = fitted.get_forecast(steps=periods)
    conf_int = forecast_result.conf_int(alpha=0.05)

    return forecast, conf_int

def moving_average_forecast(df, window=30, periods=30):
    """
    Simple moving average forecast.

    Args:
        df: DataFrame with Close prices
        window: Window size for moving average
        periods: Number of days to forecast

    Returns:
        forecast_values
    """
    prices = df['Close'].values
    ma = pd.Series(prices).rolling(window=window).mean().iloc[-1]

    # Simple forecast: extend the last moving average
    forecast = np.full(periods, ma)

    return forecast

def evaluate_forecast(y_true, y_pred):
    """
    Calculate forecast metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }

def create_forecast_plot(df, forecast_results, periods=30, save_path=None):
    """
    Create visualization comparing forecasts.

    Args:
        df: Original DataFrame
        forecast_results: Dict with forecast results
        periods: Number of forecast periods
        save_path: Optional path to save figure

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot historical data (last 365 days)
    recent = df.tail(365)
    ax.plot(recent['Date'], recent['Close'], label='Historical', color='blue', linewidth=2)

    # Generate future dates
    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')

    colors = ['orange', 'green', 'red']
    for i, (model_name, forecast_data) in enumerate(forecast_results.items()):
        forecast = forecast_data['forecast']
        conf_int = forecast_data.get('confidence', None)

        ax.plot(future_dates, forecast, label=f'{model_name} Forecast',
                color=colors[i % len(colors)], linestyle='--', linewidth=2)

        # Plot confidence interval if available
        if conf_int is not None:
            ax.fill_between(future_dates, conf_int[:, 0], conf_int[:, 1],
                           alpha=0.2, color=colors[i % len(colors)])

    ax.set_title('Gold Price Forecast Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

if __name__ == "__main__":
    print("Forecasting module loaded successfully!")
    print("\nAvailable forecasting methods:")
    print("- exponential_smoothing_forecast()")
    print("- arima_forecast()")
    print("- moving_average_forecast()")
    print("\nFor time series features:")
    print("- create_time_series_features()")
