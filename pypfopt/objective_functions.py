'''
objective_functions.py

Provides optimisation objectives for portfolio optimisation, supporting both NumPy arrays and cvxpy Variables.
'''
import logging
from typing import Union, Any

import numpy as np
import cvxpy as cp

# Module constants
DEFAULT_RISK_FREE_RATE: float = 0.02
DEFAULT_TRANSACTION_COST: float = 0.001

logger = logging.getLogger(__name__)

def _resolve_objective(
    w: Union[np.ndarray, cp.Variable],
    expr: Union[cp.Expression, float]
) -> Union[float, cp.Expression]:
    """
    Return the numeric value or cvxpy expression based on the type of 'w'.

    :param w: weight vector (NumPy array or cvxpy Variable)
    :param expr: objective expression or scalar
    :return: evaluated numeric objective or cvxpy expression
    """
    # If w is a NumPy array, we need a numeric return
    if isinstance(w, np.ndarray):
        if isinstance(expr, cp.Expression):
            try:
                val = expr.value
                if np.isscalar(val):
                    return val
                return float(np.asarray(val).item())
            except Exception as e:
                logger.error("Error resolving cvxpy expression: %s", e)
                raise
        # expr is already numeric
        return expr  # type: ignore
    # For cvxpy Variable, return the expression itself
    return expr  # type: ignore


def portfolio_variance(
    w: Union[np.ndarray, cp.Variable],
    cov_matrix: Union[np.ndarray, Any]
) -> Union[float, cp.Expression]:
    """
    Portfolio variance objective: w^T Σ w.
    """
    expr = cp.quad_form(w, cov_matrix)
    return _resolve_objective(w, expr)


def portfolio_return(
    w: Union[np.ndarray, cp.Variable],
    expected_returns: Union[np.ndarray, Any],
    negative: bool = True
) -> Union[float, cp.Expression]:
    """
    Portfolio return objective (negative for minimisation).
    R = w^T μ
    """
    sign = -1.0 if negative else 1.0
    expr = w @ expected_returns * sign
    return _resolve_objective(w, expr)


def sharpe_ratio(
    w: Union[np.ndarray, cp.Variable],
    expected_returns: Union[np.ndarray, Any],
    cov_matrix: Union[np.ndarray, Any],
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    negative: bool = True
) -> Union[float, cp.Expression]:
    """
    Portfolio Sharpe ratio objective (negative for minimisation).
    SR = (w^T μ - Rf) / sqrt(w^T Σ w)
    """
    mu = w @ expected_returns
    sigma = cp.sqrt(cp.quad_form(w, cov_matrix))
    expr = (mu - risk_free_rate) / sigma
    if negative:
        expr = -expr
    return _resolve_objective(w, expr)


def L2_reg(
    w: Union[np.ndarray, cp.Variable],
    gamma: float = 1.0
) -> Union[float, cp.Expression]:
    """
    L2 regularization: γ * ||w||^2.
    """
    expr = gamma * cp.sum_squares(w)
    return _resolve_objective(w, expr)


def quadratic_utility(
    w: Union[np.ndarray, cp.Variable],
    expected_returns: Union[np.ndarray, Any],
    cov_matrix: Union[np.ndarray, Any],
    risk_aversion: float,
    negative: bool = True
) -> Union[float, cp.Expression]:
    """
    Quadratic utility objective: μ - 0.5 * δ * w^T Σ w.
    """
    mu = w @ expected_returns
    var = cp.quad_form(w, cov_matrix)
    expr = mu - 0.5 * risk_aversion * var
    if negative:
        expr = -expr
    return _resolve_objective(w, expr)


def transaction_cost(
    w: Union[np.ndarray, cp.Variable],
    w_prev: Union[np.ndarray, Any],
    k: float = DEFAULT_TRANSACTION_COST
) -> Union[float, cp.Expression]:
    """
    Transaction cost objective: k * ||w - w_prev||_1.
    """
    expr = k * cp.norm1(w - w_prev)
    return _resolve_objective(w, expr)
