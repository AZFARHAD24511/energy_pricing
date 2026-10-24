import numpy as np
from .data_validation import LATITUDE_MAP


def make_features(long_df):
    df = long_df.copy()
    df["Latitude"] = df["Country"].map(LATITUDE_MAP)
    df = df.dropna(subset=["Latitude", "Price", "datetime"]).copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    X = df[["Latitude", "hour_sin", "hour_cos"]].astype(float)
    y = df["Price"].astype(float)
    return df, X, y
