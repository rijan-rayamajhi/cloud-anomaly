"""Zero-shot anomaly detection with MOMENT, run locally on CPU. Scored with TSB-AD metrics.
Use the 3.12 venv: .venv312/bin/python run_moment.py  (momentfm won't build on 3.14)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, pandas as pd
from pathlib import Path
from momentfm import MOMENTPipeline
from TSB_AD.evaluation.metrics import get_metrics
from data import load, all_series

RESULTS = Path(__file__).parent / "results"
W = 512
KEEP = ["VUS-PR", "AUC-PR", "Affiliation-F", "Standard-F1"]

_model = None
def model():
    global _model
    if _model is None:
        _model = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-large",
                                                model_kwargs={"task_name": "reconstruction"})
        _model.init(); _model = _model.float().eval()
    return _model


def moment_scores(values):
    """Zero-shot per-point reconstruction error. No training on these series."""
    m = model()
    v = (values - values.mean()) / (values.std() or 1.0)
    err = np.zeros(len(v))
    for i in range(0, len(v), W):
        chunk = v[i:i + W].astype("float32")
        x = np.zeros(W, dtype="float32"); x[:len(chunk)] = chunk
        xt = torch.tensor(x).reshape(1, 1, W)
        mask = torch.ones(1, W)
        with torch.no_grad():
            rec = m(x_enc=xt, input_mask=mask).reconstruction.squeeze().numpy()
        err[i:i + len(chunk)] = np.abs(chunk - rec[:len(chunk)])
    return err


if __name__ == "__main__":
    rows = []
    for name in all_series():
        df = load(name)
        if df.label.sum() == 0:
            continue
        mt = get_metrics(moment_scores(df.value.values), df.label.values)
        rows.append(dict(series=name, model="MOMENT-zeroshot", **{k: float(mt[k]) for k in KEEP}))
        print(f"{name:45} VUS-PR={float(mt['VUS-PR']):.3f}")
    out = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "moment_metrics.csv", index=False)
    print("\nMOMENT zero-shot mean:\n", out[KEEP].mean().round(3).to_string())
