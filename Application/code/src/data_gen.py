import numpy as np
from scipy.special import expit


def data_gen(n_samples, n_vars, theta, seed):
    """Generate synthetic data for PLR models"""
    np.random.seed(seed)

    cov = _get_cov_matrix(n_vars)
    mean = np.zeros(n_vars)
    X = np.random.multivariate_normal(mean, cov, n_samples)

    # m_gt = X[:, 0] ** 2 + np.sin(X[:, 1]) + 0.5 * X[:, 2]
    # g_gt = X[:, 0] ** 2 + np.cos(X[:, 1]) + 0.3 * X[:, 2]
    m_gt = X[:, 0] + 0.25 * expit(X[:, 2])
    g_gt = expit(X[:, 0]) + 0.25 * X[:, 2]
    u = np.random.normal(0, 1, n_samples)
    v = np.random.normal(0, 1, n_samples)
    d = m_gt + v
    y = theta * d + g_gt + u

    return X, y, d


def _get_cov_matrix(n):
    """Topelitz cov matrix to simulate feature correlation"""
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov[i, j] = 0.5 ** abs(i - j)
    return cov
