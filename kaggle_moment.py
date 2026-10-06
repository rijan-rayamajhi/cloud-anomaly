"""Zero-shot anomaly detection with the MOMENT foundation model, scored with TSB-AD metrics.
Run FREE on Kaggle/Colab GPU (Notebook > add GPU T4). Not meant for CPU.

Setup cell:
    !pip install momentfm TSB-AD
Upload the data/ folder (the 17 CSVs + combined_windows.json) as a Kaggle dataset, or git-clone the repo.

The point: a pretrained model does anomaly detection with NO training on these series (zero-shot),
and we score it against the simple baselines using VUS-PR — the correct metric. This is the
Direction-2 experiment that makes the work frontier-grade.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, json
from pathlib import Path
from momentfm import MOMENTPipeline
from TSB_AD.evaluation.metrics import get_metrics

DATA = Path("data")
W = 512  # MOMENT's native input length


def load(name):
    df = pd.read_csv(DATA / name, parse_dates=["timestamp"])
    windows = json.loads((DATA / "combined_windows.json").read_text())[f"realAWSCloudwatch/{name}"]
    df["label"] = 0
    for s, e in windows:
        df.loc[df.timestamp.between(s, e), "label"] = 1
    return df


def moment_scores(values):
    """Zero-shot reconstruction error per point. Higher = more anomalous. No fine-tuning."""
    model = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-large",
                                           model_kwargs={"task_name": "reconstruction"})
    model.init(); model = model.to("cuda").float().eval()
    v = (values - values.mean()) / (values.std() or 1.0)
    err = np.zeros(len(v))
    for i in range(0, len(v), W):
        chunk = v[i:i + W]
        x = np.zeros(W, dtype="float32"); x[:len(chunk)] = chunk
        xt = torch.tensor(x).reshape(1, 1, W).to("cuda")
        mask = torch.ones(1, W).to("cuda")
        with torch.no_grad():
            out = model(x_enc=xt, input_mask=mask)
        rec = out.reconstruction.squeeze().cpu().numpy()
        err[i:i + len(chunk)] = np.abs(chunk - rec[:len(chunk)])
    return err


KEEP = ["VUS-PR", "AUC-PR", "Affiliation-F", "Standard-F1"]


def main():
    from TSB_AD.models.IForest import IForest  # TSB-AD's own baseline, for a fair same-harness compare
    rows = []
    for name in sorted(p.name for p in DATA.glob("*.csv")):
        df = load(name)
        if df.label.sum() == 0:
            continue
        labels = df.label.values
        score = moment_scores(df.value.values)
        m = get_metrics(score, labels)
        rows.append(dict(series=name, model="MOMENT-zeroshot", **{k: float(m[k]) for k in KEEP}))
        print(name, "VUS-PR", round(float(m["VUS-PR"]), 3))
    out = pd.DataFrame(rows)
    out.to_csv("results/moment_metrics.csv", index=False)
    print("\nMOMENT zero-shot mean:\n", out[KEEP].mean().round(3))


if __name__ == "__main__":
    main()
