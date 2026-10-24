import pandas as pd

LATITUDE_MAP = {
    "AT": 47.5, "BE": 50.8, "CH": 46.8, "DE-LU": 51.2,
    "DK1": 55.0, "DK2": 56.0, "FI": 61.0, "FR": 46.2,
    "GB": 54.0, "NL": 52.4, "NO1": 59.9, "NO2": 63.4,
    "NO3": 58.0, "NO4": 65.0, "NO5": 58.9,
    "PL": 52.2, "SE1": 59.3, "SE2": 61.0, "SE3": 57.7, "SE4": 55.6,
}


def validate_columns(df):
    required = {"Date", "Country"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")
    if not any(c in df.columns for c in ("Price", "Hour", "00-01", "00:00")):
        hour_like = [c for c in df.columns if str(c)[:2].isdigit()]
        if not hour_like:
            raise ValueError("Could not identify hourly price columns or long-format Price/Hour columns.")


def data_quality_report(df):
    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "zones": int(df["Country"].nunique()) if "Country" in df else 0,
        "unknown_zones": sorted(set(df["Country"].dropna().astype(str)) - set(LATITUDE_MAP)) if "Country" in df else [],
    }
    return report
