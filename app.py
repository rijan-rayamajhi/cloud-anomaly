"""Dashboard: pick a server metric, see where the real problems are, and what the detector caught.
Run: .venv/bin/streamlit run app.py"""
import streamlit as st
import pandas as pd
from data import load, all_series, split
from detect import DETECTORS
from metrics_meta import label, unit, desc

st.set_page_config(page_title="Cloud Anomaly Detector", layout="wide")

NICE = {"threshold": "Fixed rule (old way)", "iforest": "Machine learning (Isolation Forest)",
        "lstm_ae": "Deep learning (LSTM)"}

st.title("🖥️ Catching problems in cloud servers before they cause failures")
st.markdown(
    "Cloud servers are usually watched by simple rules like *\"alert if CPU > 90%\"*, which miss unusual "
    "problems and raise false alarms. This dashboard runs smarter **anomaly detectors** on **real Amazon "
    "server data** where the actual failures are already known — so you can see, for each method, how many "
    "real problems it catches.")

st.sidebar.header("Try it")
name = st.sidebar.selectbox("1. Pick a server metric", all_series(), format_func=label,
                            index=all_series().index("ec2_cpu_utilization_ac20cd.csv"))
st.sidebar.caption(f"📊 {desc(name)}  ·  measured in {unit(name)}")
model = st.sidebar.selectbox("2. Pick a detection method", list(DETECTORS),
                             format_func=lambda m: NICE.get(m, m))
st.sidebar.caption("Switch the method to compare the old fixed rule against the smarter ones.")
show_fc = st.sidebar.checkbox("Show 3-hour forecast", value=False)

df = load(name)
train, test = split(df)

try:
    pred = DETECTORS[model](train, test)
except ImportError:
    st.warning(f"'{NICE.get(model, model)}' needs PyTorch (not on the hosted app) — showing Isolation Forest instead.")
    model = "iforest"
    pred = DETECTORS[model](train, test)
test = test.assign(pred=pred)
flagged = test[test.pred == 1]

st.subheader(f"{label(name)}  —  {NICE.get(model, model)}")
st.caption(f"{desc(name)}  ·  values in {unit(name)}  ·  one reading every 5 minutes")

# --- scoreboard, in plain words ---
tp = int(((test.pred == 1) & (test.label == 1)).sum())
real = int(test.label.sum())
c = st.columns(4)
c[0].metric("Data points", f"{len(df):,}", help="Total readings from this server (one every 5 minutes).")
c[1].metric("Real problems", real, help="Known failures in the test period we check against.")
c[2].metric("Caught by detector", tp, help="Real problems this method correctly flagged.")
if real:
    prec = tp / max(int(test.pred.sum()), 1)
    rec = tp / max(real, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    c[3].metric("Accuracy (F1)", f"{f1:.0%}", help="0% = useless, 100% = perfect. Higher is better.")

# --- the chart ---
base = df.set_index("timestamp")
chart = base[["value"]].rename(columns={"value": f"{label(name)} ({unit(name)})"})
chart["Real problem"] = (base.label * base.value).replace(0, None)
chart["Detector flagged"] = flagged.set_index("timestamp").value.reindex(chart.index)
st.line_chart(chart)
st.markdown(
    "**How to read this:** the blue line is the server's behaviour over time. "
    "The **Real problem** points are the actual known failures. "
    "The **Detector flagged** points are what this method raised on its own. "
    "Where they overlap, the detector caught a real failure — the goal is to catch the real ones "
    "without flagging normal behaviour.")

if show_fc:
    try:
        from prophet import Prophet
    except ImportError:
        st.info("The forecast needs Prophet, which isn't installed on the hosted app. Run locally to see it.")
        st.stop()
    from forecast import H
    hist = df.iloc[:-H].rename(columns={"timestamp": "ds", "value": "y"})[["ds", "y"]]
    m = Prophet(weekly_seasonality=False, daily_seasonality=True).fit(hist)
    fc = m.predict(m.make_future_dataframe(periods=H, freq="5min"))
    band = fc.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]].tail(300)
    band.columns = ["Predicted", "Low estimate", "High estimate"]
    st.subheader("Predicting the next 3 hours of usage")
    st.line_chart(band)

st.divider()
st.caption("Data: Numenta Anomaly Benchmark (real AWS CloudWatch metrics). "
           "Full study and code: github.com/rijan-rayamajhi/cloud-anomaly")
