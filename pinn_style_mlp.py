
import argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ship_resistance_surrogate import load, COLS

FN_IDX = 5  # position of froude in the 6 raw features


class MLP(nn.Module):
    def __init__(self, d=6, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def mono_penalty(model, x):
    """Penalise negative slope of predicted log-resistance w.r.t. (standardised) Froude number."""
    x = x.clone().requires_grad_(True)
    grad = torch.autograd.grad(model(x).sum(), x, create_graph=True)[0][:, FN_IDX]
    return torch.relu(-grad).pow(2).mean()


def train_one(Xtr, ytr, lam, seed, epochs=1500, n_col=512):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    lo, hi = Xtr.min(0).values, Xtr.max(0).values
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(Xtr), ytr)
        if lam > 0:
            idx = torch.randint(0, len(Xtr), (n_col,))
            xc = Xtr[idx].clone()
            xc[:, FN_IDX] = lo[FN_IDX] + (hi[FN_IDX] - lo[FN_IDX]) * torch.rand(n_col)  # random speeds
            loss = loss + lam * mono_penalty(model, torch.cat([Xtr, xc]))
        loss.backward(); opt.step()
    return model


def run(df, lam, n_members=5):
    X = df[COLS[:6]].values.astype(np.float32); y = np.log(df.resistance.values).astype(np.float32)
    groups = df.hull.values
    oof_mu, oof_sd = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-8; my, sy = y[tr].mean(), y[tr].std()
        Xtr = torch.tensor((X[tr] - mx) / sx); ytr = torch.tensor((y[tr] - my) / sy)
        Xte = torch.tensor((X[te] - mx) / sx)
        preds = []
        for s in range(n_members):
            m = train_one(Xtr, ytr, lam, seed=s)
            with torch.no_grad(): preds.append(m(Xte).numpy() * sy + my)
        preds = np.stack(preds)
        oof_mu[te], oof_sd[te] = preds.mean(0), preds.std(0)
    return oof_mu, oof_sd


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=None); ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args()
    df = load(a.data); y = df.resistance.values
    for lam in (0.0, a.lam):
        mu, sd = run(df, lam)
        p = np.exp(mu)
        print(f"lambda={lam:<4} RMSE={np.sqrt(mean_squared_error(y,p)):.3f} MAE={mean_absolute_error(y,p):.3f} "
              f"R2={r2_score(y,p):.3f}  95% coverage={np.mean(np.abs(np.log(y)-mu)<=1.96*sd):.1%}")
