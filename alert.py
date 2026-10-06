"""Scheduled cloud check: run detection on a series, email via Resend if anomalies found.
Used by the GitHub Actions workflow. Env: RESEND_API_KEY, ALERT_TO, ALERT_FROM."""
import os, time, json, urllib.request
from data import load
from detect import threshold, iforest
from data import split

# ponytail: CI uses the light sklearn detectors, not the LSTM (torch is heavy to install per run).
# Swap in lstm_ae here if the Action is given a torch cache.
DETECT = iforest


def check(series="ec2_cpu_utilization_ac20cd.csv"):
    t0 = time.time()
    train, test = split(load(series))
    flags = int(DETECT(train, test).sum())
    return {"series": series, "anomaly_count": flags, "detect_latency_s": round(time.time() - t0, 3)}


def email(result):
    key, to, frm = os.environ.get("RESEND_API_KEY"), os.environ.get("ALERT_TO"), os.environ.get("ALERT_FROM", "onboarding@resend.dev")
    if not (key and to):
        print("no RESEND_API_KEY/ALERT_TO set — skipping email, logging only")
        return False
    body = json.dumps({"from": frm, "to": [to],
                       "subject": f"[Cloud Anomaly] {result['anomaly_count']} anomalies in {result['series']}",
                       "text": json.dumps(result, indent=2)}).encode()
    req = urllib.request.Request("https://api.resend.com/emails", data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        print("resend:", r.status)
    return True


if __name__ == "__main__":
    import sys
    res = check(sys.argv[1] if len(sys.argv) > 1 else "ec2_cpu_utilization_ac20cd.csv")
    print(json.dumps(res))
    if res["anomaly_count"] > 0:
        email(res)
