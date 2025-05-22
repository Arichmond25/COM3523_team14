"""
The ``risk_models`` module provides functions for estimating the covariance matrix given
historical returns.

The format of the data input is the same as that in :ref:`expected-returns`.

**Currently implemented:**

- fix non-positive semidefinite matrices
- general risk matrix function, allowing you to run any risk model from one function.
- sample covariance
- semicovariance
- exponentially weighted covariance
- minimum covariance determinant
- shrunk covariance matrices:

    - manual shrinkage
    - Ledoit Wolf shrinkage
    - Oracle Approximating shrinkage

- covariance to correlation matrix
"""

import warnings
import numpy as np
import pandas as pd
from .expected_returns import returns_from_prices


def _ensure_dataframe(data, warn_msg):
    if not isinstance(data, pd.DataFrame):
        warnings.warn(warn_msg, RuntimeWarning)
        return pd.DataFrame(data)
    return data


def _ensure_returns_dataframe(prices, returns_data):
    """
    Internal helper: ensure `prices` is a DataFrame and return a DataFrame of returns.
    """
    prices_df = _ensure_dataframe(prices, "data is not in a dataframe")
    if returns_data:
        return prices_df
    return returns_from_prices(prices_df)


def _is_positive_semidefinite(matrix):
    """
    Helper to check if a matrix is positive semidefinite via Cholesky.
    """
    try:
        np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
        return True
    except np.linalg.LinAlgError:
        return False


def fix_nonpositive_semidefinite(matrix, fix_method="spectral"):
    """
    Check if `matrix` is PSD, and if not, fix it with the chosen method.
    """
    if _is_positive_semidefinite(matrix):
        return matrix

    warnings.warn(
        "The covariance matrix is non positive semidefinite. Amending eigenvalues.",
        UserWarning,
    )

    q, V = np.linalg.eigh(matrix)
    if fix_method == "spectral":
        q = np.where(q > 0, q, 0)
        fixed = V @ np.diag(q) @ V.T
    elif fix_method == "diag":
        min_eig = np.min(q)
        if min_eig < 0:
            fixed = matrix - 1.1 * min_eig * np.eye(len(matrix))
        else:
            fixed = matrix.copy()
    else:
        raise NotImplementedError(f"Method {fix_method} not implemented")

    if not _is_positive_semidefinite(fixed):
        warnings.warn("Could not fix matrix. Please try a different risk model.")

    if isinstance(matrix, pd.DataFrame):
        labels = matrix.index
        return pd.DataFrame(fixed, index=labels, columns=labels)
    return fixed


def risk_matrix(prices, method="sample_cov", returns_data=False, **kwargs):
    """
    Compute a covariance matrix using the specified risk model.
    """
    if method == "sample_cov":
        return sample_cov(prices, returns_data=returns_data, **kwargs)
    elif method == "semicovariance":
        return semicovariance(prices, returns_data=returns_data, **kwargs)
    elif method == "exp_cov":
        return exp_cov(prices, returns_data=returns_data, **kwargs)
    elif method == "min_cov_determinant":
        return min_cov_determinant(prices, returns_data=returns_data, **kwargs)
    elif method in ("ledoit_wolf", "ledoit_wolf_constant_variance"):
        return CovarianceShrinkage(prices, returns_data=returns_data, **kwargs).ledoit_wolf()
    elif method == "ledoit_wolf_single_factor":
        return CovarianceShrinkage(prices, returns_data=returns_data, **kwargs).ledoit_wolf(
            shrinkage_target="single_factor"
        )
    elif method == "ledoit_wolf_constant_correlation":
        return CovarianceShrinkage(prices, returns_data=returns_data, **kwargs).ledoit_wolf(
            shrinkage_target="constant_correlation"
        )
    elif method == "oracle_approximating":
        return CovarianceShrinkage(prices, returns_data=returns_data, **kwargs).oracle_approximating()
    else:
        raise NotImplementedError(f"Risk model {method} not implemented")


def sample_cov(prices, returns_data=False, frequency=252, fix_method="spectral"):
    """
    Calculate the annualised sample covariance matrix of asset returns.
    """
    rets = _ensure_returns_dataframe(prices, returns_data)
    cov = rets.cov() * frequency
    return fix_nonpositive_semidefinite(cov, fix_method)


def semicovariance(prices, returns_data=False, benchmark=0.000079, frequency=252, fix_method="spectral"):
    """
    Estimate the semicovariance matrix (returns < benchmark).
    """
    rets = _ensure_returns_dataframe(prices, returns_data)
    drops = np.fmin(rets - benchmark, 0)
    cov = drops.cov() * frequency
    return fix_nonpositive_semidefinite(cov, fix_method)


def _pair_exp_cov(X, Y, span=180):
    """
    Calculate the exponential covariance between two return series.
    """
    covar = (X - X.mean()) * (Y - Y.mean())
    if span < 10:
        warnings.warn("it is recommended to use a higher span, e.g 30 days")
    return covar.ewm(span=span).mean().iloc[-1]


def exp_cov(prices, returns_data=False, span=180, frequency=252, fix_method="spectral"):
    """
    Estimate the exponentially-weighted covariance matrix.
    """
    rets = _ensure_returns_dataframe(prices, returns_data)
    assets = rets.columns
    N = len(assets)
    S = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            S[i, j] = S[j, i] = _pair_exp_cov(rets.iloc[:, i], rets.iloc[:, j], span)
    cov = pd.DataFrame(S * frequency, index=assets, columns=assets)
    return fix_nonpositive_semidefinite(cov, fix_method)


def min_cov_determinant(prices, returns_data=False, frequency=252, random_state=None, fix_method="spectral"):
    """
    Calculate the minimum covariance determinant estimator.
    """
    prices_df = _ensure_dataframe(prices, "data is not in a dataframe")
    try:
        from sklearn import covariance
    except (ModuleNotFoundError, ImportError):
        raise ImportError("Please install scikit-learn via pip or poetry")

    if returns_data:
        X_df = prices_df.dropna(how="all")
    else:
        X_df = prices_df.pct_change().dropna(how="all")
    X = np.nan_to_num(X_df.values)
    raw_cov = covariance.fast_mcd(X, random_state=random_state)[1]
    cov = pd.DataFrame(raw_cov, index=prices_df.columns, columns=prices_df.columns) * frequency
    return fix_nonpositive_semidefinite(cov, fix_method)


def cov_to_corr(cov_matrix):
    """
    Convert a covariance matrix to a correlation matrix.
    """
    cov_df = _ensure_dataframe(cov_matrix, "cov_matrix is not a dataframe")
    Dinv = np.diag(1 / np.sqrt(np.diag(cov_df)))
    corr = Dinv @ cov_df.values @ Dinv
    return pd.DataFrame(corr, index=cov_df.index, columns=cov_df.index)


def corr_to_cov(corr_matrix, stdevs):
    """
    Convert a correlation matrix to a covariance matrix.
    """
    corr_df = _ensure_dataframe(corr_matrix, "cov_matrix is not a dataframe")
    cov = corr_df.values * np.outer(stdevs, stdevs)
    return pd.DataFrame(cov, index=corr_df.index, columns=corr_df.index)


class CovarianceShrinkage:
    """
    Methods for computing shrinkage estimates of the covariance matrix,
    using various targets (Ledoit–Wolf, Oracle Approximating, etc.).
    """

    def __init__(self, prices, returns_data=False, frequency=252):
        try:
            from sklearn import covariance
            self.covariance = covariance
        except (ModuleNotFoundError, ImportError):
            raise ImportError("Please install scikit-learn via pip or poetry")

        prices_df = _ensure_dataframe(prices, "data is not in a dataframe")
        self.frequency = frequency

        if returns_data:
            X_df = prices_df.dropna(how="all")
        else:
            X_df = prices_df.pct_change().dropna(how="all")
        self.X = X_df
        self.S = self.X.cov().values
        self.delta = None


    def _format_and_annualize(self, raw_cov):
        assets = self.X.columns
        cov = pd.DataFrame(raw_cov, index=assets, columns=assets) * self.frequency
        return fix_nonpositive_semidefinite(cov, fix_method="spectral")


    def shrunk_covariance(self, delta=0.2):
        """
        Manual shrinkage to the identity target.
        """
        self.delta = delta
        N = self.S.shape[1]
        mu = np.trace(self.S) / N
        F = np.identity(N) * mu
        shrunk = delta * F + (1 - delta) * self.S
        return self._format_and_annualize(shrunk)


    def ledoit_wolf(self, shrinkage_target="constant_variance"):
        """
        Ledoit–Wolf shrinkage estimator for various targets.
        """
        if shrinkage_target == "constant_variance":
            arr = np.nan_to_num(self.X.values)
            shrunk, self.delta = self.covariance.ledoit_wolf(arr)
        elif shrinkage_target == "single_factor":
            shrunk, self.delta = self._ledoit_wolf_single_factor()
        elif shrinkage_target == "constant_correlation":
            shrunk, self.delta = self._ledoit_wolf_constant_correlation()
        else:
            raise NotImplementedError(f"Shrinkage target {shrinkage_target} not recognised")
        return self._format_and_annualize(shrunk)


    def _ledoit_wolf_single_factor(self):
        """
        Helper: Ledoit–Wolf single-factor shrinkage target.
        """
        X = np.nan_to_num(self.X.values)
        t, n = X.shape
        Xm = X - X.mean(axis=0)
        xmkt = X.mean(axis=1).reshape(t, 1)

        sample = np.cov(np.append(Xm, xmkt, axis=1), rowvar=False) * (t - 1) / t
        betas = sample[:n, n].reshape(n, 1)
        varmkt = sample[n, n]
        sample = sample[:n, :n]

        F = betas @ betas.T / varmkt
        F[np.eye(n) == 1] = np.diag(sample)

        c = np.linalg.norm(sample - F, "fro") ** 2
        y = Xm ** 2
        p = 1 / t * np.sum(y.T @ y) - np.sum(sample ** 2)

        rdiag = 1 / t * np.sum(y ** 2) - np.sum(np.diag(sample) ** 2)
        z = Xm * np.tile(xmkt, (n,))
        v1 = 1 / t * (y.T @ z) - np.tile(betas, (n,)) * sample
        roff1 = (
            np.sum(v1 * np.tile(betas, (n,)).T) / varmkt
            - np.sum(np.diag(v1) * betas.T) / varmkt
        )
        v3 = 1 / t * (z.T @ z) - varmkt * sample
        roff3 = (
            np.sum(v3 * (betas @ betas.T)) / varmkt ** 2
            - np.sum(np.diag(v3).reshape(-1, 1) * betas ** 2) / varmkt ** 2
        )
        roff = 2 * roff1 - roff3
        r = rdiag + roff

        k = (p - r) / c
        delta = max(0, min(1, k / t))

        shrunk = delta * F + (1 - delta) * sample
        return shrunk, delta


    def _ledoit_wolf_constant_correlation(self):
        """
        Helper: Ledoit–Wolf constant-correlation shrinkage target.
        """
        X = np.nan_to_num(self.X.values)
        t, n = X.shape
        S = self.S

        var = np.diag(S).reshape(-1, 1)
        std = np.sqrt(var)
        _var = np.tile(var, (n, n))
        _std = np.tile(std, (n, n))
        r_bar = (np.sum(S / (_std * _std.T)) - n) / (n * (n - 1))

        F = r_bar * (_std * _std.T)
        F[np.eye(n) == 1] = var.reshape(-1)

        Xm = X - X.mean(axis=0)
        y = Xm ** 2
        pi_mat = (y.T @ y) / t - 2 * (Xm.T @ Xm) * S / t + S ** 2
        pi_hat = np.sum(pi_mat)

        term1 = (Xm ** 3).T @ Xm / t
        help_ = Xm.T @ Xm / t
        help_diag = np.diag(help_)
        term2 = np.tile(help_diag, (n, 1)).T * S
        term3 = help_ * _var
        term4 = _var * S
        theta = term1 - term2 - term3 + term4
        theta[np.eye(n) == 1] = 0
        rho_hat = (
            np.sum(np.diag(pi_mat))
            + r_bar * np.sum((1 / std) @ std.T * theta)
        )

        gamma_hat = np.linalg.norm(S - F, "fro") ** 2
        kappa_hat = (pi_hat - rho_hat) / gamma_hat
        delta = max(0.0, min(1.0, kappa_hat / t))

        shrunk = delta * F + (1 - delta) * S
        return shrunk, delta


    def oracle_approximating(self):
        """
        Oracle Approximating Shrinkage estimator.
        """
        arr = np.nan_to_num(self.X.values)
        shrunk, self.delta = self.covariance.oas(arr)
        return self._format_and_annualize(shrunk)
