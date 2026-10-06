"""Unified benchmark: all detectors on the IDENTICAL evaluation window, scored with TSB-AD metrics.
Run with the 3.12 venv (has momentfm): .venv312/bin/python benchmark.py
Outputs results/benchmark.csv + results/benchmark_summary.csv and prints the headline table."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from TSB_AD.evaluation.metrics import get_metrics
from data import load, all_series
from run_moment import moment_scores

RESULTS = Path(__file__).parent / "results"
KEEP = ["VUS-PR", "AUC-PR", "Affiliation-F", "Standard-F1", "PA-F1"]


def s_threshold(v, k=3):
    """Unsupervised z-score magnitude — the classic 'fixed rule' baseline, as a continuous score."""
    return np.abs((v - v.mean()) / (v.std() or 1.0))


def s_iforest(v):
    from sklearn.ensemble import IsolationForest
    w = 12
    s = pd.Series(v)
    feat = np.c_[v, s.rolling(w, 1).mean(), s.rolling(w, 1).std().fillna(0), s.diff().fillna(0)]
    m = IsolationForest(n_estimators=200, random_state=0).fit(feat)
    return -m.score_samples(feat)  # higher = more anomalous


MODELS = {"threshold": s_threshold, "iforest": s_iforest, "MOMENT-zeroshot": moment_scores}


def run():
    rows = []
    for name in all_series():
        df = load(name)
        if df.label.sum() == 0:
            continue
        v, y = df.value.values.astype(float), df.label.values
        for model, fn in MODELS.items():
            m = get_metrics(fn(v), y)  # identical v, identical y for every model
            rows.append(dict(series=name, model=model, **{k: float(m[k]) for k in KEEP}))
            print(f"{name:40} {model:16} VUS-PR={float(m['VUS-PR']):.3f}")
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "benchmark.csv", index=False)
    summary = df.groupby("model")[KEEP].mean().round(3).sort_values("VUS-PR", ascending=False)
    summary.to_csv(RESULTS / "benchmark_summary.csv")
    print("\n=== Mean over series (identical eval window, all models) ===")
    print(summary.to_string())
    return df, summary


if __name__ == "__main__":
    run()
