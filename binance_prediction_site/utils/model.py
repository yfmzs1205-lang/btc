"""
Model training utilities for crypto price trend prediction.

This module defines functions to build and evaluate a classifier that
predicts whether the cryptocurrency price will decline after a given
time horizon.  A random forest model from scikit-learn is used for
its robustness and ease of use.  The functions return both the
trained model and evaluation metrics.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_classifier(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """Train a random forest classifier on the provided data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature columns and a binary target column.
    feature_cols : list[str]
        Names of columns in `df` to use as model features.
    target_col : str, optional
        Name of the binary target column.  Defaults to 'target'.
    test_size : float, optional
        Fraction of data to reserve for validation.  Defaults to 0.2.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    model : RandomForestClassifier
        The trained random forest model.
    metrics : dict
        A dictionary of evaluation metrics on the validation set.
    """
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    return model, metrics


def simulate_trades(
    df: pd.DataFrame,
    model: RandomForestClassifier,
    feature_cols: list[str],
    horizon: int,
) -> Dict[str, float]:
    """Simulate a simple trading strategy based on model predictions.

    For each row in the DataFrame (except the last `horizon` rows), the
    model prediction determines whether to enter a long (predict=0) or short
    (predict=1) trade.  The trade is closed `horizon` minutes later using
    the close price at that time.  Profit is calculated as the difference
    between entry and exit prices (long trades profit when exit > entry;
    short trades profit when entry > exit).  No transaction costs are
    considered.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features and a 'close' column.  It
        should already include indicator columns and a 'target' column.
    model : RandomForestClassifier
        Trained classifier used to predict the direction.
    feature_cols : list[str]
        Feature column names used for prediction.
    horizon : int
        Number of minutes ahead to close the trade.

    Returns
    -------
    Dict[str, float]
        A dictionary containing total profit, average profit per trade,
        and win rate (fraction of profitable trades).
    """
    profits = []
    # We iterate up to len(df) - horizon to avoid indexing beyond the end
    for i in range(len(df) - horizon):
        row = df.iloc[i]
        entry_price = row["close"]
        features = row[feature_cols].values.reshape(1, -1)
        prob = model.predict_proba(features)[0, 1]
        pred = int(prob > 0.5)  # 1 => decline predicted, short trade; 0 => long
        exit_price = df.iloc[i + horizon]["close"]
        if pred == 1:
            # Short trade: profit when price declines
            profit = entry_price - exit_price
        else:
            # Long trade: profit when price increases
            profit = exit_price - entry_price
        profits.append(profit)
    total_profit = float(np.sum(profits))
    avg_profit = float(np.mean(profits)) if profits else 0.0
    win_rate = float(np.sum([1 for p in profits if p > 0]) / len(profits)) if profits else 0.0
    return {
        "total_profit": total_profit,
        "avg_profit": avg_profit,
        "win_rate": win_rate,
    }


def predict_latest(
    model: RandomForestClassifier, latest_row: pd.Series, feature_cols: list[str]
) -> Tuple[int, float]:
    """Predict the trend direction for the latest feature row.

    Parameters
    ----------
    model : RandomForestClassifier
        Trained random forest model.
    latest_row : pd.Series
        A single-row Series containing feature columns.
    feature_cols : list[str]
        List of feature column names to extract from `latest_row`.

    Returns
    -------
    int
        Predicted class label (1 if price expected to decline, 0 otherwise).
    float
        Estimated probability of the positive class (decline).
    """
    X_latest = latest_row[feature_cols].values.reshape(1, -1)
    prob = model.predict_proba(X_latest)[0, 1]
    pred = int(prob > 0.5)
    return pred, prob