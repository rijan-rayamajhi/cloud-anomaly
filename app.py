"""Dashboard: pick a series, see its metric, the detected anomalies, and the Prophet forecast band.
Run: .venv/bin/streamlit run app.py"""
import streamlit as st
import pandas as pd
from data import load, all_series, split
from detect import DETECTORS

st.set_page_config(page_title="Cloud Anomaly Monitor", layout="wide")
st.title("Cloud Anomaly & Forecast Monitor")

name = st.sidebar.selectbox("Series", all_series())
model = st.sidebar.selectbox("Detector", list(DETECTORS))
show_fc = st.sidebar.checkbox("Show forecast (slow)", value=False)

df = load(name)
train, test = split(df)

# --- detection on the test half ---
pred = DETECTORS[model](train, test) if test.label.sum() or True else None
test = test.assign(pred=pred)
flagged = test[test.pred == 1]
truth = df[df.label == 1]

st.subheader(f"{name} — {model}")
c = st.columns(4)
c[0].metric("readings", len(df))
c[1].metric("true anomalies", int(df.label.sum()))
c[2].metric("flagged (test)", int(test.pred.sum()))
if test.label.sum():
    tp = int(((test.pred == 1) & (test.label == 1)).sum())
    prec = tp / max(test.pred.sum(), 1)
    rec = tp / max(test.label.sum(), 1)
    c[3].metric("F1 (test)", f"{2 * prec * rec / max(prec + rec, 1e-9):.2f}")

chart = df.set_index("timestamp")[["value"]].rename(columns={"value": "metric"})
chart["true anomaly"] = df.set_index("timestamp").label * df.set_index("timestamp").value
chart.loc[chart["true anomaly"] == 0, "true anomaly"] = None
fl = flagged.set_index("timestamp").value
chart["flagged"] = fl.reindex(chart.index)
st.line_chart(chart)
st.caption("Line = metric. 'true anomaly' = labelled windows. 'flagged' = points this detector raised on the test half.")

if show_fc:
    from forecast import forecast_one, H
    from prophet import Prophet
    hist = df.iloc[:-H].rename(columns={"timestamp": "ds", "value": "y"})[["ds", "y"]]
    m = Prophet(weekly_seasonality=False, daily_seasonality=True).fit(hist)
    fc = m.predict(m.make_future_dataframe(periods=H, freq="5min"))
    band = fc.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]].tail(300)
    st.subheader("Forecast (last stretch + next 3h)")
    st.line_chart(band)
