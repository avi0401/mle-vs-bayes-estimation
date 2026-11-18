"""Simulated data: two Gaussian classes with known parameters (so ground truth is available)."""
import numpy as np


class GaussianMixtureProblem:
    def __init__(self, means, covs, priors, seed=None):
        self.means = [np.asarray(m, float) for m in means]
        self.covs = [np.asarray(c, float) for c in covs]
        self.priors = np.asarray(priors, float)
        self.d = len(self.means[0])
        self.rng = np.random.default_rng(seed)

    def sample_per_class(self, n_per_class):
        X, y = [], []
        for k, (m, c) in enumerate(zip(self.means, self.covs)):
            X.append(self.rng.multivariate_normal(m, c, size=n_per_class))
            y.append(np.full(n_per_class, k))
        return np.vstack(X), np.concatenate(y)

    def sample(self, n):
        y = self.rng.choice(len(self.priors), size=n, p=self.priors)
        X = np.empty((n, self.d))
        for k in range(len(self.priors)):
            idx = y == k
            X[idx] = self.rng.multivariate_normal(self.means[k], self.covs[k], size=idx.sum())
        return X, y

    def bayes_optimal_proba(self, X):
        """True posterior P(y|x) — the best any classifier can do."""
        from scipy.special import logsumexp
        from .estimators import mvn_logpdf
        lj = np.column_stack([np.log(p) + mvn_logpdf(X, m, c)
                              for p, m, c in zip(self.priors, self.means, self.covs)])
        return np.exp(lj - logsumexp(lj, axis=1, keepdims=True))


def problem_2d(seed=0):
    """Two overlapping 2-D classes with different covariances (quadratic Bayes boundary)."""
    return GaussianMixtureProblem(
        means=[[0.0, 0.0], [1.5, 1.0]],
        covs=[[[1.0, 0.5], [0.5, 1.0]], [[1.5, -0.4], [-0.4, 0.8]]],
        priors=[0.5, 0.5], seed=seed)


def problem_highdim(d=10, seed=0):
    """d-dimensional problem where the class difference is spread across all dimensions."""
    rng = np.random.default_rng(1234)
    A = rng.normal(size=(d, d)) / np.sqrt(d)
    cov0 = np.eye(d) + 0.3 * (A @ A.T)
    B = rng.normal(size=(d, d)) / np.sqrt(d)
    cov1 = np.eye(d) + 0.3 * (B @ B.T)
    mean1 = np.full(d, 1.0 / np.sqrt(d)) * 1.6
    return GaussianMixtureProblem(means=[np.zeros(d), mean1], covs=[cov0, cov1],
                                  priors=[0.5, 0.5], seed=seed)
