# Data Processing Module

import pandas as pd
import numpy as np
from datetime import datetime

def load_data(filepath):
    """
    Load and clean the gold price dataset.

    Args:
        filepath: Path to the CSV file

    Returns:
        Cleaned DataFrame
    """
    df = pd.read_csv(filepath)

    # Convert date column to datetime (format: DD-MM-YY)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y')

    return df

def add_features(df):
    """
    Add engineered features for analysis.

    Args:
        df: DataFrame with gold price data

    Returns:
        DataFrame with new features
    """
    # Create copy to avoid modifying original
    df = df.copy()

    # Add time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Quarter'] = df['Date'].dt.quarter

    # Add price-based features (if Close column exists)
    if 'Close' in df.columns:
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_MA_7'] = df['Close'].rolling(window=7).mean()
        df['Price_MA_30'] = df['Close'].rolling(window=30).mean()

    return df

def get_summary_stats(df):
    """
    Generate summary statistics.

    Args:
        df: DataFrame with gold price data

    Returns:
        Dictionary with summary statistics
    """
    stats = {
        'total_records': len(df),
        'date_range': (df['Date'].min(), df['Date'].max()),
        'summary': df.describe()
    }
    return stats

if __name__ == "__main__":
    # Example usage
    print("Data processing module loaded successfully!")
    print("Use load_data() to load your dataset")
