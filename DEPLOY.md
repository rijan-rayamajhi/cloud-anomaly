# Phase 7 — Deploy (free, no billing account)

Two free services replace GCP: **GitHub Actions** (scheduled detection + Resend email alert) and
**Streamlit Community Cloud** (hosted dashboard). Git repo is already committed locally.

## 1. Push to GitHub
```
! gh repo create cloud-anomaly --public --source=. --push
# or, if you made the repo in the browser:
! git remote add origin https://github.com/YOUR_USER/cloud-anomaly.git
! git push -u origin main
```

## 2. Resend email alerts (replaces SNS)
1. Sign up free at resend.com, create an API key (no card for the test domain).
2. In the GitHub repo → Settings → Secrets and variables → Actions, add:
   - `RESEND_API_KEY` = your key
   - `ALERT_TO` = the email to alert
   - `ALERT_FROM` = `onboarding@resend.dev` (Resend's test sender; use your own domain later)
3. Actions tab → "cloud-anomaly-check" → Run workflow. It detects and emails if anomalies.
   It also runs hourly on its own (cron in `.github/workflows/detect.yml`).

## 3. Streamlit dashboard (replaces the EC2-hosted UI)
1. Go to share.streamlit.io, connect the repo, main file `app.py`.
2. It installs `requirements.txt` and gives a public URL to show Dr. Leni.
   (The dashboard runs threshold + Isolation Forest; the LSTM needs torch, which is heavy for the
    free tier — run that one locally with `.venv/bin/streamlit run app.py`.)

## For the write-up
- Deployment: GitHub Actions (scheduled run) + Resend (email alert) + Streamlit Cloud (dashboard).
- Detection-to-result latency: ~0.4 s locally (see `alert.py` output); the Action logs it each run.
