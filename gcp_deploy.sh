#!/usr/bin/env bash
# Phase 7 — deploy to Google Cloud. Run AFTER the two manual setup steps in GCP_SETUP.md.
# Safe to re-run; it skips creation if the resource exists.
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT=cloud-anomaly-demo}"
REGION="${REGION:-us-central1}"
BUCKET="gs://${PROJECT}-data"
ALERT_EMAIL="${ALERT_EMAIL:?set ALERT_EMAIL=you@example.com}"

gcloud config set project "$PROJECT"
gcloud services enable storage.googleapis.com cloudfunctions.googleapis.com \
  monitoring.googleapis.com cloudbuild.googleapis.com run.googleapis.com

# 1. Storage: bucket for data + models (replaces S3)
gsutil ls "$BUCKET" >/dev/null 2>&1 || gsutil mb -l "$REGION" "$BUCKET"
gsutil -m cp data/*.csv data/combined_windows.json "$BUCKET/data/"
echo "Data in $BUCKET/data/"

# 2. Compute: Cloud Function runs the detector (replaces EC2/Lambda).
# Authenticated only — call it with an identity token, never public.
gcloud functions deploy anomaly-detector \
  --gen2 --runtime=python312 --region="$REGION" \
  --source=gcf --entry-point=run --trigger-http --no-allow-unauthenticated \
  --set-env-vars="BUCKET=$BUCKET" --memory=512Mi --timeout=300s

# 3. Alert: Cloud Monitoring email channel (replaces SNS)
gcloud beta monitoring channels create \
  --display-name="anomaly-email" --type=email \
  --channel-labels="email_address=$ALERT_EMAIL" 2>/dev/null || true
echo "Email channel ready for $ALERT_EMAIL — attach it to an alert policy on metric 'anomaly_count' in Console."

URL=$(gcloud functions describe anomaly-detector --region="$REGION" --gen2 --format="value(serviceConfig.uri)")
echo "Function URL: $URL"
echo "Test it (authenticated):"
echo "  curl -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" \"$URL?series=ec2_cpu_utilization_ac20cd.csv\""
