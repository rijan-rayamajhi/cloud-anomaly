"""Anomaly detectors, scored point-wise against NAB labels on the test half."""
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from data import load, all_series, split

RESULTS = __import__("pathlib").Path(__file__).parent / "results"


def threshold(train, test, k=3):
    """Fixed rule: flag |value - mean| > k*std of train (the 'CPU > X' style baseline)."""
    mu, sd = train.value.mean(), train.value.std()
    return ((test.value - mu).abs() > k * sd).astype(int)


def features(df, w=12):
    """value + 1h rolling mean/std (12 x 5min) + diff, so slow drifts and jitter are visible, not just spikes."""
    v = df.value
    return pd.DataFrame({"v": v, "mean": v.rolling(w, 1).mean(), "std": v.rolling(w, 1).std().fillna(0), "diff": v.diff().fillna(0)})


def iforest(train, test, contamination=0.05):
    """Unsupervised: fit on train (labels never seen), flag the most isolated test points."""
    from sklearn.ensemble import IsolationForest
    m = IsolationForest(n_estimators=200, contamination=contamination, random_state=0).fit(features(train))
    return (m.predict(features(test)) == -1).astype(int)


def lstm_ae(train, test, **kw):
    """Lazy wrapper so importing this module doesn't require torch (dashboard/CI run without it)."""
    from lstm_ae import lstm_ae as _impl
    return _impl(train, test, **kw)


DETECTORS = {"threshold": threshold, "iforest": iforest, "lstm_ae": lstm_ae}


def evaluate():
    rows = []
    for name in all_series():
        train, test = split(load(name))
        if test.label.sum() == 0:  # nothing to detect -> F1 undefined; skip (covers c6585a + split edge cases)
            print(f"skip {name}: no anomalies in test half")
            continue
        for model, fn in DETECTORS.items():
            p, r, f, _ = precision_recall_fscore_support(test.label, fn(train, test), average="binary", zero_division=0)
            rows.append(dict(series=name, model=model, precision=p, recall=r, f1=f))
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "metrics.csv", index=False)
    return df


if __name__ == "__main__":
    import numpy as np
    # self-check: obvious spike gets flagged, flat data does not
    tr = pd.DataFrame({"value": np.r_[np.zeros(50), np.ones(50)]})
    te = pd.DataFrame({"value": [0.5, 0.5, 100.0]})
    assert threshold(tr, te).tolist() == [0, 0, 1]
    assert iforest(tr, te)[-1] == 1

    df = evaluate()
    print(df.round(3).to_string(index=False))
    print("\nmean over series:\n", df.groupby("model")[["precision", "recall", "f1"]].mean().round(3))
