import numpy as np
import cvxpy as cp


def _objective_value(w, obj):
    """
    Return either the raw value (for np.ndarray inputs) or the cvxpy expression.
    """
    # If w is not a numpy array, assume we're building a CVXPY problem
    if not isinstance(w, np.ndarray):
        return obj

    # If obj is already a scalar, just return it
    if np.isscalar(obj):
        return obj

    # Otherwise obj is a CVXPY expression: extract its numeric value
    val = obj.value
    if np.isscalar(val):
        return val
    return val.item()


def portfolio_variance(w, cov_matrix):
    """
    Total portfolio variance (i.e., the square of volatility).
    """
    expr = cp.quad_form(w, cov_matrix)
    return _objective_value(w, expr)


def portfolio_return(w, expected_returns, negative=True):
    """
    (Negative) mean return of a portfolio.
    """
    mu = w @ expected_returns
    return _objective_value(w, -mu if negative else mu)


def sharpe_ratio(w, expected_returns, cov_matrix, risk_free_rate=0.02, negative=True):
    """
    (Negative) Sharpe ratio of a portfolio.
    """
    mu = w @ expected_returns
    sigma = cp.sqrt(cp.quad_form(w, cov_matrix))
    sharpe = (mu - risk_free_rate) / sigma
    return _objective_value(w, -sharpe if negative else sharpe)


def L2_reg(w, gamma=1):
    """
    L2 regularisation: γ · ‖w‖².
    """
    expr = gamma * cp.sum_squares(w)
    return _objective_value(w, expr)


def quadratic_utility(w, expected_returns, cov_matrix, risk_aversion, negative=True):
    """
    (Negative) quadratic utility: μ – ½·δ·wᵀΣw.
    """
    mu = w @ expected_returns
    variance = cp.quad_form(w, cov_matrix)
    util = mu - 0.5 * risk_aversion * variance
    return _objective_value(w, -util if negative else util)


def transaction_cost(w, w_prev, k=0.001):
    """
    Transaction cost: k · ∑ |w – w_prev|.
    """
    expr = k * cp.norm(w - w_prev, 1)
    return _objective_value(w, expr)
