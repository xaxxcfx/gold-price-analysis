# Model Module

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

def prepare_features(df, target_col='Close', feature_cols=None):
    """
    Prepare features for modeling.

    Args:
        df: DataFrame with features
        target_col: Name of target column
        feature_cols: List of feature columns (None for auto-selection)

    Returns:
        X, y feature and target arrays
    """
    if feature_cols is None:
        # Auto-select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_col]

    # Drop rows with NaN values
    data = df[feature_cols + [target_col]].dropna()

    X = data[feature_cols]
    y = data[target_col]

    return X, y

def train_model(X, y, model_type='random_forest'):
    """
    Train a regression model.

    Args:
        X: Feature matrix
        y: Target vector
        model_type: 'linear' or 'random_forest'

    Returns:
        Trained model and metrics dict
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Select model
    if model_type == 'linear':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }

    return model, metrics, (X_test, y_test, y_pred)

def save_model(model, filepath):
    """
    Save trained model to file.

    Args:
        model: Trained model
        filepath: Path to save model
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Load trained model from file.

    Args:
        filepath: Path to saved model

    Returns:
        Loaded model
    """
    return joblib.load(filepath)

if __name__ == "__main__":
    print("Model module loaded successfully!")
    print("Available functions:")
    print("- prepare_features()")
    print("- train_model()")
    print("- save_model()")
    print("- load_model()")
