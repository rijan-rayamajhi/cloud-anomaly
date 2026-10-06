# cloud-anomaly

**A comparative study of classical and foundation-model anomaly detection on cloud server metrics, under correct evaluation.**

🔴 **[Live dashboard](https://cloud-anomaly-rijan-rayamajhi.streamlit.app/)** · 📄 **[Full report](REPORT.md)**

Fixed-threshold monitoring misses gradual failures and floods teams with false alarms. This project
compares three detectors on real AWS CloudWatch metrics and — crucially — scores them with
threshold-independent metrics (VUS-PR), avoiding the point-adjustment protocol now known to overstate
anomaly-detection performance.

## Headline result

All models scored on the **identical** evaluation window (`benchmark.py`):

| model | VUS-PR *(correct metric)* | PA-F1 *(discredited)* |
|---|---|---|
| **MOMENT zero-shot foundation model** | **0.250** | 0.963 |
| Isolation Forest | 0.228 | 0.962 |
| Fixed threshold | 0.190 | 0.920 |

Two findings:
1. A **zero-shot foundation model** (MOMENT-1-large, no training on this data) is competitive with — and slightly beats — tuned classical baselines.
2. **PA-F1 inflates every method to ~0.95**, including the trivial threshold — reproducing the "illusion of progress" critique on independent data. The evaluation metric changes the conclusion.

See [`REPORT.md`](REPORT.md) for the full write-up and `results/fig_*.png` for figures.

## What's here

| file | purpose |
|---|---|
| `data.py` | load NAB AWS CloudWatch series + anomaly labels |
| `detect.py` | threshold, Isolation Forest, LSTM autoencoder detectors |
| `lstm_ae.py` | LSTM autoencoder (PyTorch) |
| `forecast.py` | Prophet forecasting (MAE/RMSE vs naive baseline) |
| `benchmark.py` | **unified benchmark** — all models, identical window, TSB-AD metrics |
| `run_moment.py` | zero-shot MOMENT foundation model, runs on CPU |
| `evaluate_tsb.py` | baselines scored with correct TSB-AD metrics |
| `app.py` | Streamlit dashboard (metrics, anomalies, forecast) |
| `alert.py` + `.github/workflows/detect.yml` | scheduled cloud detection + Resend email alerts |
| `REPORT.md` | the study write-up |

## Reproduce

```bash
# baselines, forecast, dashboard (Python 3.13+ ok)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python detect.py          # detection metrics
.venv/bin/python forecast.py        # forecasting metrics
.venv/bin/streamlit run app.py      # dashboard

# foundation-model benchmark (needs Python 3.12 for momentfm)
python3.12 -m venv .venv312
.venv312/bin/pip install setuptools wheel && .venv312/bin/pip install numpy pandas torch transformers scikit-learn matplotlib TSB-AD momentfm
.venv312/bin/python benchmark.py    # unified table + figures
```

## Data
14 labelled real AWS CloudWatch series from the [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) (`data/`).

## Limitations & next steps
Univariate NAB subset; modest absolute VUS-PR; one foundation model. Scales to the full
[TSB-AD](https://github.com/TheDatumOrg/TSB-AD) suite (1,070 series) — the `kaggle_moment.py` variant
targets free GPU for that.
