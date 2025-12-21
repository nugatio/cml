import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

RF_PARAMS = {
    "n_estimators": 250,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": False,
}


def dml_plr(X, y, d, n_splits, is_cross_fit, seed):
    """Double ML PLR estimator with optional k-fold cross-fitting"""
    RF_PARAMS["random_state"] = seed
    ml_m = RandomForestRegressor(**RF_PARAMS)
    ml_g = RandomForestRegressor(**RF_PARAMS)

    if is_cross_fit:
        d_res, y_res = _cross_fit(X, y, d, ml_m, ml_g, n_splits, seed)
    else:
        d_res, y_res = _overfit(X, y, d, ml_m, ml_g, seed)

    ols = LinearRegression(fit_intercept=False)
    ols.fit(d_res.reshape(-1, 1), y_res)

    return ols.coef_[0]


def _cross_fit(X, y, d, ml_m, ml_g, n_splits, seed):
    d_res = np.zeros_like(d)
    y_res = np.zeros_like(y)

    for t, v in KFold(n_splits=n_splits, random_state=seed, shuffle=True).split(X):
        X_train, X_val = X[t], X[v]
        y_train, y_val = y[t], y[v]
        d_train, d_val = d[t], d[v]

        m_hat = clone(ml_m).fit(X_train, d_train).predict(X_val)
        d_res[v] = d_val - m_hat

        g_hat = clone(ml_g).fit(X_train, y_train).predict(X_val)
        y_res[v] = y_val - g_hat

    return d_res, y_res


def _overfit(X, y, d, ml_m, ml_g, seed):
    d_res = np.zeros_like(d)
    y_res = np.zeros_like(y)

    m_hat = ml_m.fit(X, d).predict(X)
    d_res = d - m_hat

    g_hat = ml_g.fit(X, y).predict(X)
    y_res = y - g_hat

    return d_res, y_res
