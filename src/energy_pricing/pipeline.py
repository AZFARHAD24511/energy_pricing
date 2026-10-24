from pathlib import Path
import joblib
import pandas as pd

from .config import load_config
from .data_loader import load_merged_data
from .data_validation import data_quality_report
from .preprocessing import prepare_long
from .features import make_features
from .models import build_models
from .evaluation import chronological_split, regression_metrics
from .visualization import (
    save_price_over_time,
    save_feature_importance,
    save_actual_vs_predicted,
)


def run_pipeline(config_path=None):
    cfg = load_config(config_path)

    reports = Path(cfg["outputs"]["reports_dir"])
    figures = reports / "figures"
    tables = reports / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    raw = load_merged_data(cfg["data"]["merged_csv"])
    quality = data_quality_report(raw)
    print("Input data quality:", quality)

    long = prepare_long(raw)
    if long.empty:
        raise ValueError("No valid hourly observations remain after preprocessing.")

    df, X, y = make_features(long)
    if len(df) != len(X) or len(X) != len(y):
        raise RuntimeError("Feature/target alignment failed.")

    X_train, X_test, y_train, y_test, cutoff = chronological_split(
        df, X, y, float(cfg["model"].get("test_fraction", 0.30))
    )

    if X_train.empty or X_test.empty:
        raise ValueError("Chronological split produced an empty train or test set.")

    metric_rows = []

    for name, model in build_models(cfg).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        metric_rows.append({
            "model": name,
            **regression_metrics(y_test, pred)
        })

        if name == "random_forest":
            model_path = Path(cfg["outputs"]["model_path"])
            model_path.parent.mkdir(parents=True, exist_ok=True)

            joblib.dump({
                "model": model,
                "feature_names": list(X.columns),
                "cutoff": str(cutoff)
            }, model_path)

            save_feature_importance(
                model, X.columns,
                figures / "feature_importance.png"
            )

            prediction_df = pd.DataFrame({
                "datetime": df.loc[y_test.index, "datetime"].values,
                "Country": df.loc[y_test.index, "Country"].values,
                "actual": y_test.to_numpy(),
                "predicted": pred
            })

            prediction_df.to_csv(
                tables / "test_predictions.csv",
                index=False
            )

            save_actual_vs_predicted(
                prediction_df,
                figures / "actual_vs_predicted.png"
            )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(tables / "metrics.csv", index=False)

    save_price_over_time(
        df,
        figures / "mean_price_over_time.png"
    )

    print(f"Chronological test period starts at: {cutoff}")
    print(metrics_df.to_string(index=False))
    print(f"Reports saved under: {reports.resolve()}")

    return metrics_df
