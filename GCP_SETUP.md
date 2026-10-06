# Phase 7 — Google Cloud deployment

Two manual steps first (they touch billing, so run them yourself with `!`):

```
# 1. pick your user account (not the service account)
! gcloud config set account YOUR_EMAIL@gmail.com

# 2. create a dedicated project + link billing
! gcloud projects create cloud-anomaly-demo
! gcloud billing accounts list                       # copy the ACCOUNT_ID
! gcloud billing projects link cloud-anomaly-demo --billing-account=ACCOUNT_ID
```

Then one script does storage + compute + alerting:

```
! cd ~/Desktop/academic-project/cloud-anomaly
! chmod +x gcp_deploy.sh
! PROJECT=cloud-anomaly-demo ALERT_EMAIL=you@example.com ./gcp_deploy.sh
```

What it creates:
- **Cloud Storage** `gs://cloud-anomaly-demo-data` — the 17 CSVs + labels (replaces S3)
- **Cloud Function** `anomaly-detector` — authenticated only; returns anomaly_count + latency (replaces EC2/Lambda)
- **Monitoring email channel** — attach to an alert policy on `anomaly_count` in Console (replaces SNS)

The function is **not public** — call it with an identity token:
```
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" "<FUNCTION_URL>?series=ec2_cpu_utilization_ac20cd.csv"
```

AWS → GCP mapping, for the write-up: S3 → Cloud Storage, EC2/Lambda → Cloud Function, SNS → Cloud Monitoring alerts.
