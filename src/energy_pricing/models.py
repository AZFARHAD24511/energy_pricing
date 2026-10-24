from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor


def build_models(config):
    m = config["model"]
    return {
        "median_baseline": DummyRegressor(strategy="median"),
        "random_forest": RandomForestRegressor(
            n_estimators=int(m.get("n_estimators", 300)),
            max_depth=m.get("max_depth", 12),
            random_state=int(m.get("random_state", 42)),
            n_jobs=int(m.get("n_jobs", -1)),
        ),
    }
