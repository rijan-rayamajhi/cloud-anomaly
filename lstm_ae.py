"""LSTM autoencoder detector. Flags windows whose reconstruction error exceeds the train p99."""
import numpy as np, torch, torch.nn as nn

W = 48  # 4h window at 5-min cadence


def _windows(v):
    return np.stack([v[i:i + W] for i in range(len(v) - W + 1)])


class AE(nn.Module):
    def __init__(self, h=16):
        super().__init__()
        self.enc = nn.LSTM(1, h, batch_first=True)
        self.dec = nn.LSTM(h, h, batch_first=True)
        self.out = nn.Linear(h, 1)

    def forward(self, x):
        _, (hn, _) = self.enc(x)
        z = hn[-1].unsqueeze(1).repeat(1, x.size(1), 1)  # repeat latent across timesteps
        d, _ = self.dec(z)
        return self.out(d)


def lstm_ae(train, test, epochs=20, seed=0):
    torch.manual_seed(seed)
    mu, sd = train.value.mean(), train.value.std() or 1.0
    norm = lambda s: ((s.value.values - mu) / sd).astype("float32")
    Xtr = torch.tensor(_windows(norm(train)))[..., None]
    Xte = torch.tensor(_windows(norm(test)))[..., None]

    m = AE()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    lossf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(m(Xtr), Xtr); loss.backward(); opt.step()

    m.eval()
    with torch.no_grad():
        err_tr = ((m(Xtr) - Xtr) ** 2).mean((1, 2)).numpy()
        err_te = ((m(Xte) - Xte) ** 2).mean((1, 2)).numpy()
    thr = np.quantile(err_tr, 0.99)
    # window flagged -> mark its last point; first W-1 test points can't form a window -> 0
    flag = (err_te > thr).astype(int)
    out = np.zeros(len(test), dtype=int)
    out[W - 1:] = flag
    return out


if __name__ == "__main__":
    import pandas as pd
    rng = np.random.default_rng(0)
    tr = pd.DataFrame({"value": rng.normal(0, 1, 600)})
    te = pd.DataFrame({"value": np.r_[rng.normal(0, 1, 200), rng.normal(8, 1, 100)]})
    out = lstm_ae(tr, te, epochs=15)
    assert out[250:].sum() > out[:200].sum(), "anomalous tail should flag more than clean head"
    print("self-check ok, flags in tail:", int(out[250:].sum()))
