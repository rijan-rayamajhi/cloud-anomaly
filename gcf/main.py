"""Cloud Function: load a series from GCS, run detection, report anomaly count + latency."""
import os, time, json
import functions_framework


@functions_framework.http
def run(request):
    t0 = time.time()
    # ponytail: downloads from GCS per call; fine for a scheduled demo, cache the model if calls get frequent
    import pandas as pd
    from google.cloud import storage
    bucket = os.environ["BUCKET"].replace("gs://", "")
    series = request.args.get("series") or "ec2_cpu_utilization_ac20cd.csv"
    local = f"/tmp/{series}"
    storage.Client().bucket(bucket).blob(f"data/{series}").download_to_filename(local)
    df = pd.read_csv(local, parse_dates=["timestamp"])
    # threshold detection inline; the LSTM model would be loaded from GCS in a fuller build
    mu, sd = df.value.mean(), df.value.std()
    flags = int(((df.value - mu).abs() > 3 * sd).sum())
    return json.dumps({"series": series, "anomaly_count": flags,
                       "detect_latency_s": round(time.time() - t0, 3)})
