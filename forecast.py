"""Forecast next few hours of usage with Prophet; score MAE/RMSE vs a naive last-value baseline."""
import logging, warnings
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from data import load, all_series, split

RESULTS = Path(__file__).parent / "results"
H = 36  # forecast horizon: 36 x 5min = 3h


def forecast_one(df):
    """Train on all-but-last-H, predict H steps. Returns (mae, rmse, naive_mae, naive_rmse)."""
    from prophet import Prophet
    hist, future = df.iloc[:-H], df.iloc[-H:]
    m = Prophet(weekly_seasonality=False, daily_seasonality=True)
    m.fit(hist.rename(columns={"timestamp": "ds", "value": "y"})[["ds", "y"]])
    pred = m.predict(future.rename(columns={"timestamp": "ds"})[["ds"]])["yhat"].values
    actual = future.value.values
    naive = np.full(H, hist.value.iloc[-1])  # baseline: tomorrow == today's last value
    err = lambda p: (np.mean(np.abs(actual - p)), np.sqrt(np.mean((actual - p) ** 2)))
    return (*err(pred), *err(naive))


def evaluate(limit=None):
    rows = []
    names = all_series()[:limit] if limit else all_series()
    for name in names:
        df = load(name)
        mae, rmse, nmae, nrmse = forecast_one(df)
        sd = df.value.std() or 1.0
        rows.append(dict(series=name, mae=mae, rmse=rmse, naive_mae=nmae, naive_rmse=nrmse,
                         nmae=mae / sd, naive_nmae=nmae / sd))
        print(f"{name:45} mae={mae:9.2f} rmse={rmse:9.2f}  (naive mae={nmae:9.2f})")
    out = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "forecast_metrics.csv", index=False)
    wins = (out.mae < out.naive_mae).sum()
    print("\nnormalized MAE (MAE/std, lower=better)  prophet=%.3f  naive=%.3f" % (out.nmae.mean(), out.naive_nmae.mean()))
    print("Prophet beats naive on %d / %d series" % (wins, len(out)))
    return out


if __name__ == "__main__":
    # self-check: on a clean ramp, Prophet should beat naive last-value
    t = pd.date_range("2024-01-01", periods=300, freq="5min")
    df = pd.DataFrame({"timestamp": t, "value": np.arange(300, dtype=float)})
    mae, rmse, nmae, nrmse = forecast_one(df)
    assert mae < nmae, f"prophet {mae} should beat naive {nmae} on a ramp"
    print("self-check ok\n")
    evaluate()
