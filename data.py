"""Load NAB realAWSCloudwatch series with anomaly labels."""
import json
from pathlib import Path
import pandas as pd

DATA = Path(__file__).parent / "data"


def load(name):
    """Return df[timestamp, value, label] for one series; label=1 inside a labelled anomaly window."""
    df = pd.read_csv(DATA / name, parse_dates=["timestamp"])
    windows = json.loads((DATA / "combined_windows.json").read_text())[f"realAWSCloudwatch/{name}"]
    df["label"] = 0
    for start, end in windows:
        df.loc[df.timestamp.between(start, end), "label"] = 1
    return df


def all_series():
    return sorted(p.name for p in DATA.glob("*.csv"))


def split(df, frac=0.5):
    """Time-ordered split, no shuffling."""
    i = int(len(df) * frac)
    return df.iloc[:i], df.iloc[i:]


if __name__ == "__main__":
    names = all_series()
    assert len(names) == 17, names
    for n in names:
        df = load(n)
        assert df.value.notna().all() and df.timestamp.is_monotonic_increasing, n
        print(f"{n:45} rows={len(df):5} anomalous={df.label.sum():4} ({df.label.mean():.1%})")
