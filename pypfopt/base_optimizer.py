"""
The ``base_optimizer`` module houses the parent classes ``BaseOptimizer`` from which all
optimisers will inherit. ``BaseConvexOptimizer`` is the base class for all ``cvxpy`` (and ``scipy``)
optimisation.

Additionally, we define a general utility function ``portfolio_performance`` to
evaluate return and risk for a given set of portfolio weights.
"""

import collections
import json
import os
import warnings

import numpy as np
import pandas as pd
import cvxpy as cp
import scipy.optimize as sco

from . import objective_functions, exceptions


class BaseOptimizer:
    """
    Instance variables:

    - ``n_assets`` - int
    - ``tickers`` - list of str
    - ``weights`` - np.ndarray

    Public methods:

    - ``set_weights()`` creates self.weights (np.ndarray) from a weights dict
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(self, n_assets, tickers=None):
        self.n_assets = n_assets
        self.tickers = list(tickers) if tickers is not None else list(range(n_assets))
        self.weights = None

    def _make_output_weights(self, weights=None):
        """
        Turn a weight array into an OrderedDict mapping tickers → weights.
        """
        arr = self.weights if weights is None else weights
        return collections.OrderedDict(zip(self.tickers, arr))

    def set_weights(self, input_weights):
        """
        Set self.weights (np.ndarray) from a dict mapping tickers → weight.
        """
        self.weights = np.array([input_weights[t] for t in self.tickers])

    def clean_weights(self, cutoff=1e-4, rounding=5):
        """
        Zero-out any |weight| < cutoff, then round (if rounding is not None).
        """
        if self.weights is None:
            raise AttributeError("Weights not yet computed")

        w = self.weights.copy()
        w[np.abs(w) < cutoff] = 0

        if rounding is not None:
            if not isinstance(rounding, int) or rounding < 1:
                raise ValueError("rounding must be a positive integer")
            w = np.round(w, rounding)

        return self._make_output_weights(w)

    def save_weights_to_file(self, filename="weights.csv"):
        """
        Save cleaned weights to a file. Supports .csv, .json, or .txt.
        """
        clean = self.clean_weights()
        ext = os.path.splitext(filename)[1].lower().lstrip(".")

        if ext == "csv":
            pd.Series(clean).to_csv(filename, header=False)
        elif ext == "json":
            with open(filename, "w") as fp:
                json.dump(clean, fp)
        elif ext == "txt":
            with open(filename, "w") as f:
                f.write(str(dict(clean)))
        else:
            raise NotImplementedError("Only supports .txt, .json, or .csv")


class BaseConvexOptimizer(BaseOptimizer):
    """
    The BaseConvexOptimizer contains private variables for use by cvxpy.
    Public methods include `add_objective`, `add_constraint`, `convex_objective`, etc.
    """

    def __init__(self, n_assets, tickers=None, weight_bounds=(0, 1), solver=None, verbose=False):
        super().__init__(n_assets, tickers)
        self._w = cp.Variable(n_assets)
        self._objective = None
        self._additional_objectives = []
        self._constraints = []
        self._lower_bounds = None
        self._upper_bounds = None
        self._map_bounds_to_constraints(weight_bounds)

        self._solver = solver
        self._verbose = verbose

    def _map_bounds_to_constraints(self, test_bounds):
        """
        Convert user-specified bounds into CVXPY constraints.
        """
        # Per-asset bounds provided?
        if len(test_bounds) == self.n_assets and not isinstance(test_bounds[0], (float, int)):
            arr = np.array(test_bounds, dtype=float)
            self._lower_bounds = np.nan_to_num(arr[:, 0], nan=-np.inf)
            self._upper_bounds = np.nan_to_num(arr[:, 1], nan=np.inf)
        else:
            # Expect a (lower, upper) pair
            if not isinstance(test_bounds, (tuple, list)) or len(test_bounds) != 2:
                raise TypeError(
                    "test_bounds must be a pair (lower, upper) "
                    "or a collection of per-asset bounds"
                )
            low, high = test_bounds
            # Scalars or None → broadcast
            if np.isscalar(low) or low is None:
                low_val = -1.0 if low is None else low
                self._lower_bounds = np.array([low_val] * self.n_assets)
                high_val = 1.0 if high is None else high
                self._upper_bounds = np.array([high_val] * self.n_assets)
            else:
                # Arrays or array-like
                self._lower_bounds = np.nan_to_num(low, nan=-1.0)
                self._upper_bounds = np.nan_to_num(high, nan=1.0)

        self._constraints.append(self._w >= self._lower_bounds)
        self._constraints.append(self._w <= self._upper_bounds)

    def _solve_cvxpy_opt_problem(self):
        """
        Solve the assembled CVXPY problem, raise on failure.
        """
        try:
            prob = cp.Problem(cp.Minimize(self._objective), self._constraints)
            if self._solver:
                prob.solve(solver=self._solver, verbose=self._verbose)
            else:
                prob.solve(verbose=self._verbose)
        except (TypeError, cp.DCPError) as e:
            raise exceptions.OptimizationError from e

        if prob.status != "optimal":
            raise exceptions.OptimizationError

        # Round solution to avoid signed zeros
        sol = self._w.value.round(16) + 0.0
        self.weights = sol
        return self._make_output_weights()

    def add_objective(self, new_objective, **kwargs):
        self._additional_objectives.append(new_objective(self._w, **kwargs))

    def add_constraint(self, new_constraint):
        if not callable(new_constraint):
            raise TypeError("New constraint must be a callable")
        self._constraints.append(new_constraint(self._w))

    def add_sector_constraints(self, sector_mapper, sector_lower, sector_upper):
        if np.any(self._lower_bounds < 0):
            warnings.warn(
                "Sector constraints may be unreasonable with shorting allowed."
            )
        for sector, ub in sector_upper.items():
            mask = [sector_mapper[t] == sector for t in self.tickers]
            self._constraints.append(cp.sum(self._w[mask]) <= ub)
        for sector, lb in sector_lower.items():
            mask = [sector_mapper[t] == sector for t in self.tickers]
            self._constraints.append(cp.sum(self._w[mask]) >= lb)

    def convex_objective(self, custom_objective, weights_sum_to_one=True, **kwargs):
        self._objective = custom_objective(self._w, **kwargs)
        for obj in self._additional_objectives:
            self._objective += obj
        if weights_sum_to_one:
            self._constraints.append(cp.sum(self._w) == 1)
        return self._solve_cvxpy_opt_problem()

    def nonconvex_objective(
        self,
        custom_objective,
        objective_args=None,
        weights_sum_to_one=True,
        constraints=None,
        solver="SLSQP",
        initial_guess=None,
    ):
        # Prepare args
        if not isinstance(objective_args, tuple):
            objective_args = (objective_args,)
        # Build SciPy bounds
        bound_array = np.vstack((self._lower_bounds, self._upper_bounds)).T
        bounds = list(map(tuple, bound_array))
        # Default starting point
        if initial_guess is None:
            initial_guess = np.array([1 / self.n_assets] * self.n_assets)
        # Build constraints
        final_constraints = []
        if weights_sum_to_one:
            final_constraints.append({"type": "eq", "fun": lambda x: np.sum(x) - 1})
        if constraints:
            final_constraints += constraints
        res = sco.minimize(
            custom_objective,
            x0=initial_guess,
            args=objective_args,
            method=solver,
            bounds=bounds,
            constraints=final_constraints,
        )
        self.weights = res.x
        return self._make_output_weights()


def portfolio_performance(
    weights, expected_returns, cov_matrix, verbose=False, risk_free_rate=0.02
):
    """
    Calculate expected return, volatility, and Sharpe ratio for a set of weights.
    """
    # Normalize input weights
    if isinstance(weights, dict):
        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        elif isinstance(cov_matrix, pd.DataFrame):
            tickers = list(cov_matrix.columns)
        else:
            tickers = list(range(len(expected_returns)))
        w_arr = np.zeros(len(tickers))
        for i, t in enumerate(tickers):
            w_arr[i] = weights.get(t, 0.0)
        if w_arr.sum() == 0:
            raise ValueError("Weights add to zero, or ticker names don't match")
    elif weights is not None:
        w_arr = np.asarray(weights)
    else:
        raise ValueError("Weights is None")

    sigma = np.sqrt(objective_functions.portfolio_variance(w_arr, cov_matrix))

    if expected_returns is not None:
        mu = objective_functions.portfolio_return(
            w_arr, expected_returns, negative=False
        )
        sharpe = objective_functions.sharpe_ratio(
            w_arr,
            expected_returns,
            cov_matrix,
            risk_free_rate=risk_free_rate,
            negative=False,
        )
        if verbose:
            print(f"Expected annual return: {100 * mu:.1f}%")
            print(f"Annual volatility: {100 * sigma:.1f}%")
            print(f"Sharpe Ratio: {sharpe:.2f}")
        return mu, sigma, sharpe
    else:
        if verbose:
            print(f"Annual volatility: {100 * sigma:.1f}%")
        return None, sigma, None
