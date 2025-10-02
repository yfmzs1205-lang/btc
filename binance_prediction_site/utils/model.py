"""
Model training utilities for crypto price trend classification.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# 原来的“批量训练”随机森林留着，供离线一次性训练
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# ✅ 新增：用于在线增量学习
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# -------------------------
# 批量训练（保持你的原功能）
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
# ✅ 增量学习（最简单可用）
# -------------------------
def init_incremental_model() -> "Pipeline":
    """
    返回一个可 partial_fit 的流水线：StandardScaler + SGDClassifier(logistic)
    """
    clf = SGDClassifier(loss="log_loss", learning_rate="optimal", alpha=1e-4, random_state=42)
    pipe = make_pipeline(StandardScaler(with_mean=True, with_std=True), clf)
    return pipe

def partial_fit_step(model, X_chunk: np.ndarray, y_chunk: np.ndarray):
    """
    对模型执行一次增量更新。如果是第一次，需要传入 classes=[0,1]。
    Pipeline 会把 partial_fit 递交给各步（StandardScaler、SGDClassifier）。
    """
    # 第一次需要 classes
    needs_init = not hasattr(model, "classes_") and not hasattr(model[-1], "classes_")
    if needs_init:
        model.partial_fit(X_chunk, y_chunk, classes=np.array([0, 1], dtype=int))
    else:
        model.partial_fit(X_chunk, y_chunk)

def predict_proba_safe(model, X: np.ndarray) -> np.ndarray:
    """
    取上涨(1)的概率；若模型不支持 predict_proba，退化为 decision_function 的sigmoid。
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    import numpy as np
    from scipy.special import expit
    s = model.decision_function(X).reshape(-1, 1)
    p1 = expit(s)
    return np.hstack([1 - p1, p1])
