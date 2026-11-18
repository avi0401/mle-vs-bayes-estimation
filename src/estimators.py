"""
Parameter estimators and the generative classifiers built on them.

  * MLEGaussianClassifier   - plug-in QDA: class means/covariances by maximum likelihood,
                              class priors by relative frequency.
  * BayesGaussianClassifier - full Bayesian QDA: Normal-Inverse-Wishart (NIW) prior on each
                              class's (mu, Sigma), Dirichlet prior on class priors. Predictions
                              use the posterior predictive (a multivariate Student-t), so
                              parameter uncertainty is integrated out instead of ignored.
"""
import numpy as np
from scipy.special import gammaln, logsumexp


# ----------------------------------------------------------------------------- densities
def mvn_logpdf(X, mu, Sigma):
    """Log density of a multivariate normal, evaluated row-wise on X."""
    d = X.shape[1]
    L = np.linalg.cholesky(Sigma)
    z = np.linalg.solve(L, (X - mu).T)                 # (d, n)
    maha = np.sum(z ** 2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2 * np.pi) + logdet + maha)


def mvt_logpdf(X, mu, Scale, df):
    """Log density of a multivariate Student-t (location mu, scale matrix Scale, df)."""
    d = X.shape[1]
    L = np.linalg.cholesky(Scale)
    z = np.linalg.solve(L, (X - mu).T)
    maha = np.sum(z ** 2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return (gammaln((df + d) / 2) - gammaln(df / 2)
            - 0.5 * d * np.log(df * np.pi) - 0.5 * logdet
            - 0.5 * (df + d) * np.log1p(maha / df))


# ------------------------------------------------------------------- single-class estimators
def mle_gaussian(X, ridge=1e-6):
    """MLE of (mu, Sigma). Sigma uses 1/n (biased). A tiny ridge keeps it invertible
    when n <= d; without it MLE QDA is undefined at very small sample sizes."""
    n, d = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu
    Sigma = Xc.T @ Xc / n + ridge * np.eye(d)
    return mu, Sigma


class NIWPrior:
    """Normal-Inverse-Wishart prior:  Sigma ~ IW(Psi0, nu0),  mu | Sigma ~ N(mu0, Sigma/kappa0)."""

    def __init__(self, d, mu0=None, kappa0=0.01, nu0=None, prior_cov=None):
        self.d = d
        self.mu0 = np.zeros(d) if mu0 is None else np.asarray(mu0, float)
        self.kappa0 = kappa0
        self.nu0 = d + 2 if nu0 is None else nu0           # smallest nu with a finite mean of Sigma
        prior_cov = np.eye(d) if prior_cov is None else np.asarray(prior_cov, float)
        # choose Psi0 so that the prior mean E[Sigma] = Psi0/(nu0-d-1) equals prior_cov
        self.Psi0 = prior_cov * (self.nu0 - d - 1)

    def posterior(self, X):
        n = X.shape[0]
        xbar = X.mean(axis=0)
        Xc = X - xbar
        S = Xc.T @ Xc
        kn = self.kappa0 + n
        nun = self.nu0 + n
        mun = (self.kappa0 * self.mu0 + n * xbar) / kn
        diff = (xbar - self.mu0)[:, None]
        Psin = self.Psi0 + S + (self.kappa0 * n / kn) * (diff @ diff.T)
        return NIWPosterior(mun, kn, nun, Psin)


class NIWPosterior:
    def __init__(self, mu, kappa, nu, Psi):
        self.mu, self.kappa, self.nu, self.Psi = mu, kappa, nu, Psi
        self.d = len(mu)

    @property
    def mean_mu(self):
        return self.mu

    @property
    def mean_Sigma(self):
        return self.Psi / (self.nu - self.d - 1)

    def predictive_params(self):
        """Posterior predictive p(x|D) = Student-t(df, mu_n, Scale)."""
        df = self.nu - self.d + 1
        Scale = self.Psi * (self.kappa + 1) / (self.kappa * df)
        return self.mu, Scale, df


# ------------------------------------------------------------------------------ classifiers
class MLEGaussianClassifier:
    name = "MLE (plug-in QDA)"

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n = len(y)
        self.params_ = [mle_gaussian(X[y == c]) for c in self.classes_]
        self.log_prior_ = np.log([np.sum(y == c) / n for c in self.classes_])
        return self

    def log_joint(self, X):
        return np.column_stack([lp + mvn_logpdf(X, mu, S)
                                for lp, (mu, S) in zip(self.log_prior_, self.params_)])

    def predict_proba(self, X):
        lj = self.log_joint(X)
        return np.exp(lj - logsumexp(lj, axis=1, keepdims=True))

    def predict(self, X):
        return self.classes_[np.argmax(self.log_joint(X), axis=1)]


class BayesGaussianClassifier(MLEGaussianClassifier):
    name = "Bayesian (posterior predictive)"

    def __init__(self, kappa0=0.01, nu0=None, prior_cov=None, mu0=None, alpha=1.0):
        self.kappa0, self.nu0, self.prior_cov, self.mu0, self.alpha = kappa0, nu0, prior_cov, mu0, alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        d = X.shape[1]
        # mu0 may be one vector shared by all classes, or one row per class
        mu0 = None if self.mu0 is None else np.asarray(self.mu0, float)
        per_class = mu0 is not None and mu0.ndim == 2
        self.posteriors_ = []
        for k, c in enumerate(self.classes_):
            m0 = mu0[k] if per_class else mu0
            prior = NIWPrior(d, mu0=m0, kappa0=self.kappa0, nu0=self.nu0, prior_cov=self.prior_cov)
            self.posteriors_.append(prior.posterior(X[y == c]))
        counts = np.array([np.sum(y == c) for c in self.classes_], float)
        K = len(self.classes_)
        # Dirichlet(alpha) posterior predictive for the class label
        self.log_prior_ = np.log((counts + self.alpha) / (counts.sum() + K * self.alpha))
        return self

    def log_joint(self, X):
        cols = []
        for lp, post in zip(self.log_prior_, self.posteriors_):
            mu, Scale, df = post.predictive_params()
            cols.append(lp + mvt_logpdf(X, mu, Scale, df))
        return np.column_stack(cols)


class BayesMAPPlugInClassifier(BayesGaussianClassifier):
    """Ablation: Bayesian *posterior-mean* parameters plugged into a Gaussian.
    Separates the effect of the prior (shrinkage) from that of predictive averaging."""
    name = "Bayes posterior-mean plug-in"

    def log_joint(self, X):
        return np.column_stack([lp + mvn_logpdf(X, p.mean_mu, p.mean_Sigma)
                                for lp, p in zip(self.log_prior_, self.posteriors_)])
