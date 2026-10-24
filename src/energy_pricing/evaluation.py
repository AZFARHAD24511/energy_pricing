import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def chronological_split(df, X, y, test_fraction=0.30):
    timestamps = df["datetime"].dropna().sort_values().unique()
    if len(timestamps) < 3:
        raise ValueError("At least three distinct timestamps are required for a chronological split.")
    split_idx = max(1, min(len(timestamps) - 1, int(len(timestamps) * (1 - test_fraction))))
    cutoff = timestamps[split_idx]
    train_mask = df["datetime"] < cutoff
    test_mask = df["datetime"] >= cutoff
    return X.loc[train_mask], X.loc[test_mask], y.loc[train_mask], y.loc[test_mask], cutoff
