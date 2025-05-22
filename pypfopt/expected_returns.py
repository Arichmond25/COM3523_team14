"""
The ``expected_returns`` module provides functions for estimating the expected returns of
assets, required for mean-variance optimization. All outputs are annualized returns,
and inputs can be either prices or returns (depending on the ``returns_data`` flag).
"""

import warnings
import pandas as pd
import numpy as np


def _ensure_prices_dataframe(prices):
    """
    Internal helper to ensure inputs are DataFrames. Emits the same
    RuntimeWarning as before if conversion is needed.
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not in a dataframe", RuntimeWarning)
        return pd.DataFrame(prices)
    return prices


def returns_from_prices(prices, log_returns=False):
    """
    Calculate (daily) returns from price series.
    :param prices: DataFrame of prices, each column a ticker and each row a date.
    :param log_returns: bool, if True compute log returns.
    :return: DataFrame of returns (drops rows that are all NaN).
    """
    rets = prices.pct_change()
    if log_returns:
        rets = np.log(1 + rets)
    return rets.dropna(how="all")


def log_returns_from_prices(prices):
    """
    Deprecated: calculate log returns. Use `returns_from_prices(prices, log_returns=True)` instead.
    """
    warnings.warn(
        "log_returns_from_prices is deprecated. Please use returns_from_prices(prices, log_returns=True)",
        UserWarning,
    )
    return returns_from_prices(prices, log_returns=True)


def prices_from_returns(returns, log_returns=False):
    """
    Convert a Series/DataFrame of returns into pseudo-prices (all starting at 1).
    :param returns: DataFrame of returns.
    :param log_returns: bool, if True interpret `returns` as log-returns.
    :return: DataFrame of pseudo-prices (cumprod of 1 + returns).
    """
    if log_returns:
        returns = np.exp(returns)
    pseudo = 1 + returns
    # ensure first row is exactly 1
    pseudo.iloc[0] = 1
    return pseudo.cumprod()


def return_model(prices, method="mean_historical_return", **kwargs):
    """
    Dispatcher for different return-estimation models.
    :param prices: DataFrame of prices (or returns if returns_data=True).
    :param method: one of "mean_historical_return", "ema_historical_return", "capm_return".
    :raises NotImplementedError: if method not recognized.
    """
    if method == "mean_historical_return":
        return mean_historical_return(prices, **kwargs)
    elif method == "ema_historical_return":
        return ema_historical_return(prices, **kwargs)
    elif method == "capm_return":
        return capm_return(prices, **kwargs)
    else:
        raise NotImplementedError(f"Return model {method} not implemented")


def mean_historical_return(prices, returns_data=False, compounding=True, frequency=252):
    """
    Calculate annualized mean historical return.
    :param prices: DataFrame of prices (or returns if returns_data=True).
    :param returns_data: bool, if True `prices` is actually returns.
    :param compounding: bool, geometric vs. arithmetic.
    :param frequency: periods per year.
    :return: Series of annualized returns.
    """
    prices = _ensure_prices_dataframe(prices)
    rets = prices if returns_data else returns_from_prices(prices)
    if compounding:
        return (1 + rets).prod() ** (frequency / rets.count()) - 1
    return rets.mean() * frequency


def ema_historical_return(prices, returns_data=False, compounding=True, span=500, frequency=252):
    """
    Calculate annualized exponentially-weighted mean return.
    :param span: span for the EMA.
    """
    prices = _ensure_prices_dataframe(prices)
    rets = prices if returns_data else returns_from_prices(prices)
    ew = rets.ewm(span=span).mean().iloc[-1]
    if compounding:
        return (1 + ew) ** frequency - 1
    return ew * frequency


def capm_return(
    prices,
    market_prices=None,
    returns_data=False,
    risk_free_rate=0.02,
    compounding=True,
    frequency=252,
):
    """
    Estimate returns via the CAPM: R = Rf + beta * (R_m - Rf)
    :param market_prices: DataFrame of benchmark prices (or returns if returns_data=True).
    """
    prices = _ensure_prices_dataframe(prices)

    # Prepare asset & market returns
    if returns_data:
        asset_rets = prices
        mkt_rets = market_prices
    else:
        asset_rets = returns_from_prices(prices)
        mkt_rets = (
            market_prices if market_prices is not None else None
        )
        if mkt_rets is not None:
            mkt_rets = returns_from_prices(mkt_rets)

    # Use equally-weighted mean if no benchmark passed
    if mkt_rets is None:
        asset_rets["mkt"] = asset_rets.mean(axis=1)
    else:
        mkt_rets.columns = ["mkt"]
        asset_rets = asset_rets.join(mkt_rets, how="left")

    # Compute betas
    cov = asset_rets.cov()
    betas = cov["mkt"] / cov.loc["mkt", "mkt"]
    betas = betas.drop("mkt")

    # Compute average market return
    if compounding:
        mkt_mean = (1 + asset_rets["mkt"]).prod() ** (frequency / asset_rets["mkt"].count()) - 1
    else:
        mkt_mean = asset_rets["mkt"].mean() * frequency

    # Apply CAPM formula
    return risk_free_rate + betas * (mkt_mean - risk_free_rate)
