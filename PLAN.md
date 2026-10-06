# Intelligent Cloud-Based Predictive Analytics and Anomaly Detection System

Goal: learn normal behaviour from cloud metrics, flag anomalies, forecast usage.
Supervisor: Dr. Leni (scope to confirm).

## Layout (flat on purpose; split files only when one gets painful)
```
cloud-anomaly/
  data/          NAB realAWSCloudwatch CSVs + combined_windows.json (labels)
  results/       metrics.csv, plots
  detect.py      threshold + Isolation Forest + LSTM autoencoder, scored with NAB labels
  forecast.py    Prophet (LSTM only if Prophet underperforms)
  app.py         Streamlit dashboard
  alert.py       SNS email when a point is flagged
  requirements.txt
```

## Phases
1. **Data** — download NAB `realAWSCloudwatch` (17 series) + label windows. Time-based train/test split, no shuffling.
2. **Baseline** — fixed threshold (mean + 3σ of train). Record P/R/F1.
3. **Isolation Forest** — sklearn, features = value + rolling mean/std. Record P/R/F1.
4. **LSTM autoencoder** — Keras, window 48, flag reconstruction error > train p99. Record P/R/F1.
5. **Forecast** — Prophet, next few hours. Record MAE/RMSE vs naive last-value baseline.
6. **Dashboard** — Streamlit: series, anomalies marked, forecast band.
7. **AWS** — S3 for data/models, one EC2 (or Lambda) running detect on a schedule, SNS email. Log detection-to-alert latency.
8. **Write-up** — results table → fill the letter's one sentence: "LSTM improved F1 from X to Y over thresholding".

## Results to record (results/metrics.csv)
| model | precision | recall | F1 |   + forecast MAE/RMSE   + alert latency (s)

## Skipped (add when needed)
- Docker/CI/config files — one person, one machine.
- Separate src/ package, tests folder — one assert self-check per script instead.
- Kafka/streaming — replay CSVs as "live" in the dashboard.
- Auto-remediation, multi-cloud — out of scope per spec.

## Open questions for Dr. Leni
- Is NAB acceptable, or must data come from a live AWS account?
- Is Prophet fine for forecasting, or LSTM required?
- Point-wise F1 or NAB window-based scoring?
