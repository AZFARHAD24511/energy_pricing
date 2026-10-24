import pandas as pd
from energy_pricing.evaluation import chronological_split


def test_split_keeps_later_timestamps_in_test():
    dates = pd.date_range("2024-01-01", periods=10, freq="h")
    df = pd.DataFrame({"datetime": dates})
    X = pd.DataFrame({"x": range(10)})
    y = pd.Series(range(10))
    Xtr, Xte, ytr, yte, cutoff = chronological_split(df, X, y, .3)
    assert Xtr.index.max() < Xte.index.min()
    assert len(Xte) > 0
