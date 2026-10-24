from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_price_over_time(df, output_path):
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    daily = df.groupby(df["datetime"].dt.date)["Price"].mean()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(pd.to_datetime(daily.index), daily.values, linewidth=1)
    ax.set(title="Mean hourly electricity price over time", xlabel="Date", ylabel="Price (source units)")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def save_feature_importance(model, feature_names, output_path):
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    importance.plot.barh(ax=ax)
    ax.set(title="Random Forest feature importance", xlabel="Importance", ylabel="Feature")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
