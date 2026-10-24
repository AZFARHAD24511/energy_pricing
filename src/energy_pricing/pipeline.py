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
from .visualization import save_price_over_time, save_feature_importance


def run_pipeline(config_path=None):
    cfg = load_config(config_path)
    reports = Path(cfg["outputs"]["reports_dir"])
    figures, tables = reports / "figures", reports / "tables"
    figures.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)

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
        df, X, y, float(cfg["model"].get("test_fraction", .30))
    )
    if X_train.empty or X_test.empty:
        raise ValueError("Chronological split produced an empty train or test set.")

    metric_rows, fitted = [], {}
    for name, model in build_models(cfg).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metric_rows.append({"model": name, **regression_metrics(y_test, pred)})
        fitted[name] = model
        if name == "random_forest":
            joblib.dump({"model": model, "feature_names": list(X.columns), "cutoff": str(cutoff)}, cfg["outputs"]["model_path"])
            save_feature_importance(model, X.columns, figures / "feature_importance.png")
            pd.DataFrame({"datetime": df.loc[y_test.index, "datetime"].values,
                          "Country": df.loc[y_test.index, "Country"].values,
                          "actual": y_test.values, "predicted": pred}).to_csv(tables / "test_predictions.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(tables / "metrics.csv", index=False)
    save_price_over_time(df, figures / "mean_price_over_time.png")
    print(f"Chronological test period starts at: {cutoff}")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(f"Reports saved under: {reports.resolve()}")
    return pd.DataFrame(metric_rows)
