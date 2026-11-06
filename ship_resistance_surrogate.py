
import argparse, os, urllib.request, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
SEED = 0
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"
COLS = ["lcb", "cp", "l_disp", "b_draught", "l_beam", "froude", "resistance"]
GEOM = COLS[:5]


def load(path=None):
    if path is None:
        path = "yacht_hydrodynamics.data"
        if not os.path.exists(path):
            print(f"Downloading {URL}")
            urllib.request.urlretrieve(URL, path)
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS).dropna()
    assert len(df) > 0 and (df.resistance > 0).all()
    # Hull ID = unique combination of the 5 geometry parameters (each hull is run at many speeds)
    key = df[GEOM].round(3).astype(str).agg("|".join, axis=1)
    df["hull"] = pd.factorize(key)[0]
    return df


#  features 
def add_physics_features(X):
    """Resistance grows steeply with Froude number (wave-making), so give models Fn powers + log."""
    X = X.copy()
    X["fn2"] = X.froude ** 2
    X["fn4"] = X.froude ** 4
    X["log_fn"] = np.log(X.froude)
    return X


# models (all trained on log(resistance))
def make_models(n_feat, fn_idx):
    mono = [0] * n_feat
    mono[fn_idx] = 1  # resistance must not decrease as Froude number increases
    return {
        "Poly-Ridge (deg 2)": make_pipeline(StandardScaler(), PolynomialFeatures(2), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "Poly-Ridge (deg 3)": make_pipeline(StandardScaler(), PolynomialFeatures(3), RidgeCV(alphas=np.logspace(-3, 3, 13))),
        "GBM (unconstrained)": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=SEED),
        "GBM (monotonic in Fn)": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, monotonic_cst=mono, random_state=SEED),
        "Gaussian process (ARD)": make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(C(1.0) * Matern(length_scale=np.ones(n_feat), nu=2.5) + WhiteKernel(1e-2),
                                     normalize_y=True, n_restarts_optimizer=2, random_state=SEED)),
    }


def predict_with_std(model, X):
    if "Gaussian process (ARD)" in getattr(model, "_name", ""):
        gp = model[-1]
        return gp.predict(model[:-1].transform(X), return_std=True)
    return model.predict(X), None


def cross_validate(X, y_log, groups, splitter, name_filter=None, features_note=""):
    """Returns out-of-fold predictions (log space) and GP std per model."""
    results, stds = {}, {}
    fn_idx = list(X.columns).index("froude")
    for name in make_models(X.shape[1], fn_idx):
        oof, sd = np.zeros(len(y_log)), np.full(len(y_log), np.nan)
        for tr, te in splitter.split(X, y_log, groups):
            model = make_models(X.shape[1], fn_idx)[name]
            model._name = name
            model.fit(X.iloc[tr], y_log[tr])
            mu, s = predict_with_std(model, X.iloc[te])
            oof[te] = mu
            if s is not None: sd[te] = s
        results[name], stds[name] = oof, sd
    return results, stds


def score(y_true, y_log_pred):
    y_pred = np.exp(y_log_pred)
    return dict(RMSE=np.sqrt(mean_squared_error(y_true, y_pred)),
                MAE=mean_absolute_error(y_true, y_pred),
                R2=r2_score(y_true, y_pred),
                RMSE_log=np.sqrt(mean_squared_error(np.log(y_true), y_log_pred)))


def main(args):
    from scipy.stats import wilcoxon
    df = load(args.data)
    sizes = df.groupby("hull").size()
    print(f"{len(df)} samples, {df.hull.nunique()} distinct hull forms; rows per hull: "
          f"min={sizes.min()}, median={int(sizes.median())}, max={sizes.max()}")
    if df.hull.nunique() > 0.5 * len(df):
        print("WARNING: almost every row is its own 'hull' -> grouping is not working; "
              "leave-one-hull-out would really be leave-one-out. Do not trust the honest-CV numbers.")
    y = df.resistance.values
    y_log = np.log(y)
    X = add_physics_features(df[COLS[:6]])
    groups = df.hull.values

    rows = []
    fmt = lambda s_: "  ".join(f"{a}={b:.3f}" for a, b in s_.items())

    N_REP = 3
    print(f"\n== Random 10-fold CV, mean over {N_REP} repeats (LEAKY: same hull in train and test) ==")
    rnd_all = []
    for rep in range(N_REP):
        r, _ = cross_validate(X, y_log, groups, KFold(10, shuffle=True, random_state=rep))
        rnd_all.append(r)
    for k in rnd_all[0]:
        sc = [score(y, r[k]) for r in rnd_all]
        mean = {m: np.mean([d[m] for d in sc]) for m in sc[0]}
        sd = {m: np.std([d[m] for d in sc]) for m in sc[0]}
        rows.append(dict(split="random_kfold_mean", model=k, **mean, **{f"{m}_sd": v for m, v in sd.items()}))
        print(f"{k:28s} " + fmt(mean) + f"   (RMSE sd across repeats={sd['RMSE']:.3f})")
    rnd = rnd_all[0]

    print("\n== Leave-one-hull-out CV (HONEST: predicting unseen hull forms) ==")
    loho, sds = cross_validate(X, y_log, groups, LeaveOneGroupOut())
    for k, v in loho.items():
        s_ = score(y, v); rows.append(dict(split="leave_one_hull_out", model=k, **s_)); print(f"{k:28s} " + fmt(s_))

    gp_name = "Gaussian process (ARD)"
    sd = sds[gp_name]
    cover = np.mean(np.abs(y_log - loho[gp_name]) <= 1.96 * sd)
    print(f"\nGP 95% interval coverage on unseen hulls: {cover:.1%} (ideal ~95%)")

    # paired comparison: does the monotonic constraint help, hull by hull?
    hulls = df.hull.unique()
    per_hull_mae = lambda p: np.array([np.mean(np.abs(y[groups == h] - np.exp(p[groups == h]))) for h in hulls])
    a, b = per_hull_mae(loho["GBM (unconstrained)"]), per_hull_mae(loho["GBM (monotonic in Fn)"])
    try: pval = wilcoxon(a, b).pvalue
    except ValueError: pval = float("nan")
    print(f"\nMonotonic vs unconstrained GBM (per-hull MAE): monotonic better on {(b < a).sum()}/{len(hulls)} hulls, "
          f"median diff={np.median(a - b):+.4f}, Wilcoxon p={pval:.3f}")

    pd.DataFrame(rows).to_csv("results.csv", index=False)

    #  plots 
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a_, (title, res) in zip(ax, [("Random split (leaky)", rnd), ("Leave-one-hull-out (honest)", loho)]):
        a_.scatter(y, np.exp(res[gp_name]), s=10, alpha=.6)
        lim = [0, y.max() * 1.05]; a_.plot(lim, lim, "k--", lw=1)
        a_.set_xlabel("Actual residuary resistance"); a_.set_ylabel("Predicted"); a_.set_title(f"GP: {title}")
    plt.tight_layout(); plt.savefig("pred_vs_actual.png", dpi=150); plt.close()

    hull = hulls[0]
    m = (df.hull == hull).values
    o = np.argsort(df.froude.values[m])
    fn = df.froude.values[m][o]
    mu = loho[gp_name][m][o]; s_ = sd[m][o]
    plt.figure(figsize=(6, 4.5))
    plt.plot(fn, y[m][o], "ko-", label="Measured")
    plt.plot(fn, np.exp(mu), "r-", label="GP (hull unseen in training)")
    plt.fill_between(fn, np.exp(mu - 1.96 * s_), np.exp(mu + 1.96 * s_), color="r", alpha=.2, label="95% interval")
    plt.xlabel("Froude number"); plt.ylabel("Residuary resistance"); plt.legend(); plt.tight_layout()
    plt.savefig("resistance_curve_unseen_hull.png", dpi=150); plt.close()
    print("\nSaved results.csv, pred_vs_actual.png, resistance_curve_unseen_hull.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data", default=None)
    main(p.parse_args())