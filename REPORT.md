# A Comparative Study of Classical and Foundation-Model Anomaly Detection on Cloud Server Metrics, Under Correct Evaluation

**Author:** Tanishq · **Supervisor:** Dr. Leni · **Status:** complete

## Abstract
Operations teams monitor cloud servers with fixed thresholds that miss gradual or unusual
failures. We compare three anomaly detectors on real AWS CloudWatch metrics — a fixed-threshold
baseline, an Isolation Forest, and a **zero-shot time-series foundation model (MOMENT-1-large)** —
evaluated on an identical window with threshold-independent metrics (VUS-PR), avoiding the
point-adjustment protocol now known to overstate performance. The zero-shot foundation model,
with no training on the target data, matches or slightly exceeds the tuned classical baselines,
and we show that the widely used PA-F1 metric inflates every method to near-perfect scores,
reproducing the "illusion of progress" reported in the recent benchmarking literature.

## 1. Problem
Fixed-threshold monitoring ("alert if CPU > 90%") produces false alarms and misses slow drift.
We ask: (a) do learned detectors beat fixed thresholds on real cloud metrics, (b) can a pretrained
foundation model detect anomalies *zero-shot*, and (c) how much does the evaluation metric itself
change the conclusion?

## 2. Data
14 labelled real AWS CloudWatch series (CPU, disk-write, network-in, ELB request count, RDS CPU)
from the Numenta Anomaly Benchmark. Each point is labelled anomalous/normal from NAB's windows.
Series with no anomalies are excluded from scoring.

## 3. Methods
- **Threshold** — z-score magnitude, the classical fixed rule, as a continuous score.
- **Isolation Forest** — on value + 1h rolling mean/std + first difference.
- **MOMENT-zeroshot** — MOMENT-1-large reconstruction error; **no fine-tuning** on these series.
- **Evaluation** — TSB-AD metrics. Headline is **VUS-PR** (threshold-independent, temporal-aware).
  We deliberately avoid point-adjusted F1 (PA-F1); §5 shows why.
- All models are scored on the **identical series and labels** (`benchmark.py`), so differences
  are the method, not the evaluation window.

## 4. Results
Mean over series (see `results/benchmark_summary.csv`, `fig_vuspr.png`):

| model | VUS-PR | AUC-PR | Standard-F1 | PA-F1 |
|---|---|---|---|---|
| **MOMENT-zeroshot** | **0.250** | 0.217 | 0.289 | 0.963 |
| Isolation Forest | 0.228 | 0.198 | 0.277 | 0.962 |
| threshold | 0.190 | 0.158 | 0.241 | 0.920 |

Learned detectors beat the fixed threshold, and a zero-shot foundation model — trained on no
cloud data at all — is the strongest, though the margin over Isolation Forest is small.

## 5. The evaluation matters more than the model
The PA-F1 column rates every method at 0.92–0.96, including the trivial threshold — near-perfect,
and almost indistinguishable (see `fig_illusion.png`). Across the five metrics, the *same*
predictions move by ~0.6 in score. Under PA-F1 one would wrongly conclude the problem is solved.
This reproduces, on independent data, the benchmarking critique that point-adjustment creates an
illusion of progress.

## 6. Conclusion
On correctly-scored real cloud metrics: (1) learned detection beats fixed thresholds; (2) a
zero-shot foundation model is competitive with tuned classical methods without any target-domain
training, supporting foundation models as a practical direction for operations monitoring; and
(3) the choice of evaluation metric changes the ranking and can manufacture apparent success, so
threshold-independent metrics are essential.

## 7. Limitations & next steps
Univariate NAB subset (14 series); VUS-PR values are modest in absolute terms; a single foundation
model. Natural extensions: the full TSB-AD suite (1,070 series), multivariate detection, and
characterising exactly when foundation models beat classical methods.

## Reproduce
```
python3 -m venv .venv312 --python=python3.12 ; .venv312/bin/pip install -r requirements.txt momentfm TSB-AD
.venv312/bin/python benchmark.py      # table + CSVs
```
Artifacts: `results/benchmark.csv`, `results/benchmark_summary.csv`, `results/fig_vuspr.png`,
`results/fig_illusion.png`.
