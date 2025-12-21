from dml import RF_PARAMS
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def naive_plr(X, y, d, seed):
    """Naive PLR estimator using random forest"""
    X_aux, X_main, y_aux, y_main, d_aux, d_main = train_test_split(
        X, y, d, test_size=0.5, random_state=seed
    )
    RF_PARAMS["random_state"] = seed

    ml_g = RandomForestRegressor(**RF_PARAMS)
    g_hat = ml_g.fit(X_aux, y_aux).predict(X_main)
    y_res = y_main - g_hat

    ols = LinearRegression()
    ols.fit(d_main.reshape(-1, 1), y_res)

    return ols.coef_[0]
