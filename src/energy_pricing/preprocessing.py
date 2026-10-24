import re
import pandas as pd
from .data_validation import validate_columns

ID_COLUMNS = {"Date", "Country", "Baseload", "Peakload", "Latitude", "datetime", "Hour", "Price"}


def _hour_label_to_int(value):
    text = str(value).strip()
    match = re.search(r"(\d{1,2})", text)
    if not match:
        return None
    hour = int(match.group(1))
    return hour if 0 <= hour <= 23 else None


def prepare_long(raw):
    """Convert merged wide data or already-long data to one row per zone-hour."""
    validate_columns(raw)
    df = raw.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Price" in df.columns and "Hour" in df.columns:
        long = df[["Date", "Country", "Hour", "Price"]].copy()
        long["hour"] = long["Hour"].map(_hour_label_to_int)
    else:
        hour_cols = [c for c in df.columns if c not in {"Date", "Country", "Baseload", "Peakload"}]
        long = df.melt(id_vars=["Date", "Country"], value_vars=hour_cols,
                       var_name="Hour", value_name="Price")
        long["hour"] = long["Hour"].map(_hour_label_to_int)
    long["Price"] = pd.to_numeric(long["Price"], errors="coerce")
    long["Country"] = long["Country"].astype("string").str.strip()
    long = long.dropna(subset=["Date", "Country", "Price", "hour"]).copy()
    long["hour"] = long["hour"].astype(int)
    long["datetime"] = long["Date"].dt.normalize() + pd.to_timedelta(long["hour"], unit="h")
    long = long.sort_values(["datetime", "Country"]).reset_index(drop=True)
    return long[["Date", "Country", "Hour", "hour", "Price", "datetime"]]
