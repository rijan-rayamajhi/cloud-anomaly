"""Score our detectors with TSB-AD's metrics (VUS-PR etc.) instead of the broken point-wise F1.
Demonstrates how much the metric choice alone moves the score — the 'illusion of progress' problem.
Foundation-model detectors (MOMENT/Chronos) are added from the Kaggle notebook; this file runs the
CPU baselines locally."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from TSB_AD.evaluation.metrics import get_metrics
from data import load, all_series, split
from detect import threshold, iforest

RESULTS = Path(__file__).parent / "results"
KEEP = ["VUS-PR", "AUC-PR", "Affiliation-F", "PA-F1", "Standard-F1"]  # correct ... to discredited
BASELINES = {"threshold": threshold, "iforest": iforest}


def evaluate():
    rows = []
    for name in all_series():
        tr, te = split(load(name))
        if te.label.sum() == 0:
            continue
        for model, fn in BASELINES.items():
            score = np.asarray(fn(tr, te), dtype=float)
            m = get_metrics(score, te.label.values)
            rows.append(dict(series=name, model=model, **{k: float(m[k]) for k in KEEP}))
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "tsb_metrics.csv", index=False)
    summary = df.groupby("model")[KEEP].mean().round(3)
    print(summary.to_string())
    print("\nMetric spread for iforest (min..max across the 5 metrics), per series:")
    ifo = df[df.model == "iforest"]
    spread = (ifo[KEEP].max(1) - ifo[KEEP].min(1))
    print(f"  mean spread = {spread.mean():.3f}  (same prediction, score moves this much by metric choice)")
    return df


if __name__ == "__main__":
    evaluate()
