from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_price_over_time(df, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    daily = df.groupby(df["datetime"].dt.date)["Price"].mean()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(pd.to_datetime(daily.index), daily.values, linewidth=1)
    ax.set(
        title="Mean hourly electricity price over time",
        xlabel="Date",
        ylabel="Price (source units)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_feature_importance(model, feature_names, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values()

    fig, ax = plt.subplots(figsize=(7, 4))
    importance.plot.barh(ax=ax)
    ax.set(
        title="Random Forest feature importance",
        xlabel="Importance",
        ylabel="Feature"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_actual_vs_predicted(predictions, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    actual = pd.to_numeric(
        predictions["actual"], errors="coerce"
    ).to_numpy(dtype=float)

    predicted = pd.to_numeric(
        predictions["predicted"], errors="coerce"
    ).to_numpy(dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        raise ValueError("No valid actual/predicted observations to plot.")

    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r2 = (
        1.0 - float(np.sum(errors ** 2)) / denominator
        if denominator > 0 else float("nan")
    )

    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual, predicted, alpha=0.45, s=14)
    ax.plot([lower, upper], [lower, upper], linestyle="--")

    ax.set(
        title="Actual vs Predicted Electricity Prices (Test Set)",
        xlabel="Actual price (source units)",
        ylabel="Predicted price (source units)"
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")

    metrics_text = (
        f"Observations: {len(actual):,}\n"
        f"MAE: {mae:.3f}\n"
        f"RMSE: {rmse:.3f}\n"
        f"R²: {r2:.3f}"
    )
    ax.text(
        0.04, 0.96, metrics_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "alpha": 0.85}
    )

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    print(f"Actual-vs-predicted figure saved: {path.resolve()}")
    print(
        f"Test observations: {len(actual)}, "
        f"MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}"
    )
