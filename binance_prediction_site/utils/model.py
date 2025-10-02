"""
Model training utilities for crypto price trend classification.
Supports:
- Batch training: RandomForest (for baseline metrics)
- Incremental learning: SGDClassifier + StandardScaler with manual partial_fit wrapper
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# -------------------------
# Batch training (baseline)
# -------------------------
def train_classifier(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    X = df[feature_cols].values
    y = df[target_col].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=random_state
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }
    return model, metrics

# -------------------------
# Incremental learning
# -------------------------
class _IncModel:
    """
    A tiny wrapper for incremental learning:
    - StandardScaler updated via partial_fit
    - SGDClassifier (logistic loss) updated via partial_fit
    API:
        .partial_fit(X_chunk, y_chunk)
        .predict_proba(X)
    """
    def __init__(self):
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.clf = SGDClassifier(
            loss="log_loss",
            learning_rate="optimal",
            alpha=1e-4,
            random_state=42,
        )
        self._initialized = False

    def partial_fit(self, X_chunk: np.ndarray, y_chunk: np.ndarray):
        # update scaler statistics, then transform and update classifier
        self.scaler.partial_fit(X_chunk)
        Xs = self.scaler.transform(X_chunk)
        if not self._initialized:
            self.clf.partial_fit(Xs, y_chunk, classes=np.array([0, 1], dtype=int))
            self._initialized = True
        else:
            self.clf.partial_fit(Xs, y_chunk)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        # SGDClassifier with log_loss 提供 predict_proba
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(Xs)
        # 兜底：sigmoid(decision_function)
        s = self.clf.decision_function(Xs).reshape(-1, 1)
        p1 = 1.0 / (1.0 + np.exp(-s))
        return np.hstack([1 - p1, p1])

def init_incremental_model():
    """Return a fresh incremental model wrapper."""
    return _IncModel()

def partial_fit_step(model, X_chunk: np.ndarray, y_chunk: np.ndarray):
    """
    One incremental update step.
    The 'model' here is the _IncModel above (keeps scaler + classifier).
    """
    model.partial_fit(X_chunk, y_chunk)

def predict_proba_safe(model, X: np.ndarray) -> np.ndarray:
    """
    Get P(y=1) safely for either batch or incremental models.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    # For RandomForest etc. it exists; for our wrapper we also implemented it.
    # Add final fallback just in case:
    if hasattr(model, "decision_function"):
        s = model.decision_function(X).reshape(-1, 1)
        p1 = 1.0 / (1.0 + np.exp(-s))
        return np.hstack([1 - p1, p1])
    # Worst-case uniform probability
    return np.repeat([[0.5, 0.5]], X.shape[0], axis=0)
