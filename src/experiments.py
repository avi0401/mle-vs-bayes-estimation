"""All experiments. Each function saves its figure(s) to `out` and returns a results dict."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from .data import problem_2d, problem_highdim
from .estimators import (MLEGaussianClassifier, BayesGaussianClassifier,
                         BayesMAPPlugInClassifier, mle_gaussian, NIWPrior)
from .metrics import ALL_METRICS, reliability_curve

COLORS = {"MLE": "#d1495b", "Bayes": "#00798c", "BayesPlugIn": "#edae49", "Optimal": "#555555"}
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def _models(**bayes_kw):
    return {"MLE": MLEGaussianClassifier(),
            "Bayes": BayesGaussianClassifier(**bayes_kw),
            "BayesPlugIn": BayesMAPPlugInClassifier(**bayes_kw)}


# =========================================================== 1. pure parameter estimation
def exp_parameter_estimation(out, n_trials=2000, seed=0):
    rng = np.random.default_rng(seed)
    ns = np.array([1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200])
    res = {"n": ns.tolist()}

    # (a) Bernoulli p with Beta(a,b) prior
    p_true, a, b = 0.7, 2.0, 2.0
    mse_mle, mse_bayes = [], []
    for n in ns:
        k = rng.binomial(n, p_true, size=n_trials)
        mse_mle.append(np.mean((k / n - p_true) ** 2))
        mse_bayes.append(np.mean(((k + a) / (n + a + b) - p_true) ** 2))
    res["bernoulli"] = {"mle": mse_mle, "bayes": mse_bayes}

    # (b) Gaussian mean, known sigma, N(mu0, tau^2) prior — plus 95% interval coverage
    mu_true, sigma, mu0, tau = 1.0, 2.0, 0.0, 1.5
    mse_mle_g, mse_bayes_g, cov_mle, cov_bayes = [], [], [], []
    for n in ns:
        x = rng.normal(mu_true, sigma, size=(n_trials, n))
        xbar = x.mean(axis=1)
        prec = 1 / tau ** 2 + n / sigma ** 2
        post_mean = (mu0 / tau ** 2 + n * xbar / sigma ** 2) / prec
        post_sd = np.sqrt(1 / prec)
        mse_mle_g.append(np.mean((xbar - mu_true) ** 2))
        mse_bayes_g.append(np.mean((post_mean - mu_true) ** 2))
        se = sigma / np.sqrt(n)
        cov_mle.append(np.mean(np.abs(xbar - mu_true) < 1.96 * se))
        cov_bayes.append(np.mean(np.abs(post_mean - mu_true) < 1.96 * post_sd))
    res["gauss_mean"] = {"mle": mse_mle_g, "bayes": mse_bayes_g,
                         "coverage_mle": cov_mle, "coverage_bayes": cov_bayes}

    # (c) 2-D covariance matrix (Frobenius error) — MLE vs NIW posterior mean
    prob = problem_2d(seed=seed)
    Sig_true = prob.covs[0]
    prior = NIWPrior(2, kappa0=0.01)
    ns_c = ns[ns >= 3]
    fe_mle, fe_bayes = [], []
    for n in ns_c:
        e1, e2 = [], []
        for _ in range(n_trials // 4):
            X = rng.multivariate_normal(prob.means[0], Sig_true, size=n)
            e1.append(np.linalg.norm(mle_gaussian(X, ridge=0)[1] - Sig_true))
            e2.append(np.linalg.norm(prior.posterior(X).mean_Sigma - Sig_true))
        fe_mle.append(np.mean(np.square(e1)))
        fe_bayes.append(np.mean(np.square(e2)))
    res["covariance"] = {"n": ns_c.tolist(), "mle": fe_mle, "bayes": fe_bayes}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].loglog(ns, mse_mle, "o-", c=COLORS["MLE"], label="MLE  k/n")
    ax[0].loglog(ns, mse_bayes, "s-", c=COLORS["Bayes"], label="Posterior mean (Beta(2,2) prior)")
    ax[0].set(title=f"Bernoulli p (true p={p_true})", xlabel="n", ylabel="MSE")
    ax[1].loglog(ns, mse_mle_g, "o-", c=COLORS["MLE"], label="MLE  x̄")
    ax[1].loglog(ns, mse_bayes_g, "s-", c=COLORS["Bayes"], label=f"Posterior mean (N({mu0},{tau}²) prior)")
    ax[1].loglog(ns, sigma ** 2 / ns, "k:", lw=1, label="σ²/n (Cramér–Rao bound)")
    ax[1].set(title=f"Gaussian mean (true μ={mu_true}, σ={sigma})", xlabel="n", ylabel="MSE")
    ax[2].loglog(ns_c, fe_mle, "o-", c=COLORS["MLE"], label="MLE  S/n")
    ax[2].loglog(ns_c, fe_bayes, "s-", c=COLORS["Bayes"], label="NIW posterior mean")
    ax[2].set(title="2-D covariance matrix", xlabel="n", ylabel="E‖Σ̂ − Σ‖²_F")
    for a_ in ax:
        a_.legend(fontsize=8)
    fig.suptitle("Parameter estimation error: MLE vs Bayesian estimates", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "01_parameter_estimation_mse.png"))
    plt.close(fig)

    # posterior evolution for the coin — how belief concentrates as data arrive
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = np.linspace(0, 1, 400)
    flips = rng.random(200) < p_true
    for n, shade in zip([0, 1, 5, 20, 200], np.linspace(0.25, 1, 5)):
        k = flips[:n].sum()
        ax.plot(xs, stats.beta.pdf(xs, a + k, b + n - k), c=plt.cm.viridis(shade),
                label=f"n={n}, heads={k}" + (f", MLE={k / n:.2f}" if n else ""))
    ax.axvline(p_true, c="k", ls="--", lw=1, label="true p")
    ax.set(title="Posterior over p sharpens as data arrives (Beta(2,2) prior)",
           xlabel="p", ylabel="posterior density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "02_posterior_evolution.png"))
    plt.close(fig)
    return res


# ============================================================ 2. classifier learning curves
def _learning_curves(problem_fn, ns, n_trials, n_test, seed, **bayes_kw):
    prob = problem_fn(seed=seed)
    Xte, yte = prob.sample(n_test)
    opt = prob.bayes_optimal_proba(Xte)
    optimal = {m: f(yte, opt) for m, f in ALL_METRICS.items()}
    names = list(_models().keys())
    raw = {nm: {m: np.zeros((len(ns), n_trials)) for m in ALL_METRICS} for nm in names}
    for i, n in enumerate(ns):
        for t in range(n_trials):
            Xtr, ytr = prob.sample_per_class(n)
            for nm, model in _models(**bayes_kw).items():
                p = model.fit(Xtr, ytr).predict_proba(Xte)
                for m, f in ALL_METRICS.items():
                    raw[nm][m][i, t] = f(yte, p)
    return raw, optimal


def _plot_curves(raw, optimal, ns, title, path):
    labels = {"MLE": "MLE (plug-in)", "Bayes": "Bayesian (posterior predictive)",
              "BayesPlugIn": "Bayes posterior-mean plug-in"}
    titles = {"accuracy": "Test accuracy  ↑", "log_loss": "Test log-loss  ↓",
              "brier": "Brier score  ↓", "ece": "Expected calibration error  ↓"}
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
    for ax, m in zip(axes, ALL_METRICS):
        for nm, r in raw.items():
            vals = r[m]
            if m == "log_loss":
                med = np.median(vals, axis=1)        # MLE log-loss has heavy outliers at tiny n
                lo, hi = np.percentile(vals, [25, 75], axis=1)
            else:
                med = vals.mean(axis=1)
                se = vals.std(axis=1) / np.sqrt(vals.shape[1])
                lo, hi = med - 1.96 * se, med + 1.96 * se
            ls = "--" if nm == "BayesPlugIn" else "-"
            ax.plot(ns, med, ls, marker="o", ms=4, c=COLORS[nm], label=labels[nm])
            ax.fill_between(ns, lo, hi, color=COLORS[nm], alpha=0.15)
        if m != "ece":
            ax.axhline(optimal[m], c=COLORS["Optimal"], ls=":", label="Bayes-optimal (true params)")
        ax.set_xscale("log")
        ax.set(title=titles[m] + (" (median, IQR)" if m == "log_loss" else ""),
               xlabel="training samples per class")
        if m == "log_loss":
            ax.set_ylim(optimal[m] * 0.95, min(ax.get_ylim()[1], optimal[m] * 2.2))
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _summarise(raw, optimal, ns):
    rows = []
    for i, n in enumerate(ns):
        for nm, r in raw.items():
            row = {"n_per_class": int(n), "model": nm}
            for m in ALL_METRICS:
                row[m] = float(r[m][i].mean())
                row[m + "_median"] = float(np.median(r[m][i]))
                row[m + "_std"] = float(r[m][i].std())
            rows.append(row)
    return {"rows": rows, "optimal": optimal}


def exp_learning_curves_2d(out, n_trials=200, n_test=20000, seed=0):
    ns = np.array([3, 4, 5, 8, 12, 20, 35, 60, 100, 200, 500])
    raw, opt = _learning_curves(problem_2d, ns, n_trials, n_test, seed)
    _plot_curves(raw, opt, ns, "2-D Gaussian classes: classifier performance vs training size",
                 os.path.join(out, "03_learning_curves_2d.png"))
    return _summarise(raw, opt, ns)


def exp_learning_curves_highdim(out, d=10, n_trials=100, n_test=20000, seed=0):
    ns = np.array([d + 1, d + 3, 15, 20, 30, 50, 80, 150, 300, 600])
    raw, opt = _learning_curves(lambda seed: problem_highdim(d=d, seed=seed), ns,
                                n_trials, n_test, seed)
    _plot_curves(raw, opt, ns, f"{d}-D Gaussian classes: where MLE struggles (n close to d)",
                 os.path.join(out, "04_learning_curves_highdim.png"))
    return _summarise(raw, opt, ns)


# =================================================================== 3. decision boundaries
def exp_decision_boundaries(out, seed=3):
    prob = problem_2d(seed=seed)
    gx, gy = np.meshgrid(np.linspace(-4, 5.5, 300), np.linspace(-3.5, 4.5, 300))
    G = np.column_stack([gx.ravel(), gy.ravel()])
    p_opt = prob.bayes_optimal_proba(G)[:, 1].reshape(gx.shape)
    sizes = [4, 15, 200]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for j, n in enumerate(sizes):
        Xtr, ytr = prob.sample_per_class(n)
        for i, model in enumerate([MLEGaussianClassifier(), BayesGaussianClassifier()]):
            ax = axes[i, j]
            p = model.fit(Xtr, ytr).predict_proba(G)[:, 1].reshape(gx.shape)
            cf = ax.contourf(gx, gy, p, levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.75)
            ax.contour(gx, gy, p, levels=[0.5], colors="k", linewidths=2)
            ax.contour(gx, gy, p_opt, levels=[0.5], colors="lime", linewidths=1.5, linestyles="--")
            ax.contour(gx, gy, p, levels=[0.1, 0.9], colors="k", linewidths=0.6, linestyles=":")
            ax.scatter(*Xtr[ytr == 0].T, c="#1f4e9c", edgecolor="w", s=28 if n < 50 else 8)
            ax.scatter(*Xtr[ytr == 1].T, c="#b2182b", edgecolor="w", s=28 if n < 50 else 8)
            ax.set_title(f"{model.name}\nn = {n} per class", fontsize=10)
            ax.grid(False)
    fig.colorbar(cf, ax=axes, shrink=0.6, label="P(class 1 | x)")
    fig.suptitle("Decision boundaries (black = learned P=0.5, dotted = P=0.1/0.9, "
                 "green dashed = Bayes-optimal)", fontweight="bold")
    fig.savefig(os.path.join(out, "05_decision_boundaries.png"), bbox_inches="tight")
    plt.close(fig)


# ===================================================================== 4. calibration
def exp_calibration(out, n_small=5, n_trials=300, n_test=4000, seed=1):
    prob = problem_2d(seed=seed)
    Xte, yte = prob.sample(n_test)
    preds = {"MLE": [], "Bayes": []}
    for _ in range(n_trials):
        Xtr, ytr = prob.sample_per_class(n_small)
        preds["MLE"].append(MLEGaussianClassifier().fit(Xtr, ytr).predict_proba(Xte))
        preds["Bayes"].append(BayesGaussianClassifier().fit(Xtr, ytr).predict_proba(Xte))
    y_all = np.tile(yte, n_trials)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot([0, 1], [0, 1], "k:", label="perfectly calibrated")
    res = {}
    for nm, label in [("MLE", "MLE (plug-in)"), ("Bayes", "Bayesian (posterior predictive)")]:
        P = np.vstack(preds[nm])
        mp, fp = reliability_curve(y_all, P, n_bins=15)
        ax[0].plot(mp, fp, "o-", c=COLORS[nm], label=label)
        conf = P.max(axis=1)
        ax[1].hist(conf, bins=30, range=(0.5, 1), alpha=0.55, color=COLORS[nm], label=label, density=True)
        res[nm] = {"mean_confidence": float(conf.mean()),
                   "accuracy": float((P.argmax(1) == y_all).mean()),
                   "frac_conf_above_0.99": float((conf > 0.99).mean())}
    ax[0].set(title=f"Reliability diagram (n = {n_small} per class, {n_trials} trainings)",
              xlabel="predicted P(class 1)", ylabel="observed frequency of class 1")
    ax[1].set(title="Distribution of prediction confidence", xlabel="max predicted probability",
              ylabel="density")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "06_calibration_small_n.png"))
    plt.close(fig)
    return res


# ================================================================== 5. prior sensitivity
def exp_prior_sensitivity(out, n_trials=150, n_test=10000, seed=2):
    """Vary prior strength kappa0 (= number of 'pseudo-observations' backing the prior mean)
    with a per-class prior mean that is either CORRECT (true class means) or WRONG
    (the two classes' means swapped)."""
    prob = problem_2d(seed=seed)
    Xte, yte = prob.sample(n_test)
    ns = np.array([3, 5, 10, 20, 50, 100, 300])
    kappas = [0.01, 1, 10, 50]
    true_means = np.array(prob.means)
    settings = {"correct prior: μ₀ₖ = true class means": true_means,
                "wrong prior: μ₀ₖ = the OTHER class's mean": true_means[::-1].copy()}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    res = {}
    for ax, (title, mu0) in zip(axes, settings.items()):
        mle_acc = []
        for n in ns:
            accs = []
            for _ in range(n_trials):
                Xtr, ytr = prob.sample_per_class(n)
                accs.append(ALL_METRICS["accuracy"](yte, MLEGaussianClassifier().fit(Xtr, ytr).predict_proba(Xte)))
            mle_acc.append(np.mean(accs))
        ax.plot(ns, mle_acc, "o-", c=COLORS["MLE"], lw=2.5, label="MLE")
        res[title] = {"MLE": mle_acc}
        for kc, col in zip(kappas, plt.cm.viridis(np.linspace(0.1, 0.85, len(kappas)))):
            acc = []
            for n in ns:
                accs = []
                for _ in range(n_trials):
                    Xtr, ytr = prob.sample_per_class(n)
                    m = BayesGaussianClassifier(kappa0=kc, mu0=mu0)
                    accs.append(ALL_METRICS["accuracy"](yte, m.fit(Xtr, ytr).predict_proba(Xte)))
                acc.append(np.mean(accs))
            ax.plot(ns, acc, "s--", c=col, label=f"Bayes κ₀={kc}")
            res[title][f"kappa0={kc}"] = acc
        ax.set_xscale("log")
        ax.set(title=title, xlabel="training samples per class", ylabel="test accuracy")
        ax.legend(fontsize=8)
    fig.suptitle("Prior sensitivity: a strong prior helps if right, hurts if wrong; "
                 "either way the data eventually win", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "07_prior_sensitivity.png"))
    plt.close(fig)
    res["n"] = ns.tolist()
    return res
