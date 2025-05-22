import logging
from typing import Optional, Union

import pandas as pd
import numpy as np

# Module constants
DEFAULT_FREQUENCY = 252

logger = logging.getLogger(__name__)


def _ensure_dataframe(data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    """
    Ensure the input data is a pandas DataFrame. If an ndarray is provided, convert it.
    Raises a ValueError if conversion is not possible.

    :param data: Input data as DataFrame or ndarray
    :return: DataFrame
    """
    if isinstance(data, pd.DataFrame):
        return data
    try:
        return pd.DataFrame(data)
    except Exception as e:
        logger.error("Failed to convert data to DataFrame: %s", e)
        raise ValueError("Input data must be a pandas DataFrame or convertible ndarray.")


def returns_from_prices(
    prices: Union[pd.DataFrame, np.ndarray],
    log_returns: bool = False
) -> pd.DataFrame:
    """
    Calculate period-over-period returns from price data.

    :param prices: Asset prices (e.g., daily) as a DataFrame or ndarray
    :param log_returns: If True, calculate log returns; otherwise, simple returns
    :return: Returns DataFrame
    """
    prices_df = _ensure_dataframe(prices)
    returns = prices_df.pct_change()
    if log_returns:
        returns = np.log1p(returns)
    return returns.dropna(how="all")


def prices_from_returns(
    returns: Union[pd.DataFrame, np.ndarray],
    log_returns: bool = False
) -> pd.DataFrame:
    """
    Reconstruct pseudo-prices from returns, assuming initial value of 1.

    :param returns: Returns DataFrame or ndarray
    :param log_returns: If True, interpret returns as log changes
    :return: Pseudo-prices DataFrame
    """
    ret_df = _ensure_dataframe(returns)
    if log_returns:
        ret_df = np.expm1(ret_df)
    prices = (1 + ret_df).cumprod()
    prices.iloc[0] = 1
    return prices


def return_model(
    prices: Union[pd.DataFrame, np.ndarray],
    method: str = "mean_historical_return",
    **kwargs
) -> pd.Series:
    """
    Compute an estimate of future annual returns based on the specified method.

    :param prices: Asset prices or returns (if returns_data=True)
    :param method: One of 'mean_historical_return', 'ema_historical_return', 'capm_return'
    :raises ValueError: if the method is not recognized
    :return: Annualized returns as a Series
    """
    methods = {
        "mean_historical_return": mean_historical_return,
        "ema_historical_return": ema_historical_return,
        "capm_return": capm_return,
    }
    func = methods.get(method)
    if func is None:
        raise ValueError(f"Return model '{method}' is not implemented.")
    return func(prices, **kwargs)


def mean_historical_return(
    prices: Union[pd.DataFrame, np.ndarray],
    returns_data: bool = False,
    compounding: bool = True,
    frequency: int = DEFAULT_FREQUENCY
) -> pd.Series:
    """
    Annualized mean historical return.

    :param prices: Prices or returns data
    :param returns_data: If True, 'prices' is actually returns
    :param compounding: If True, compute compounded annual return
    :param frequency: Periods per year
    :return: Series of annual returns
    """
    data = _ensure_dataframe(prices)
    returns = data if returns_data else returns_from_prices(data)

    if compounding:
        compounded = (1 + returns).prod() ** (frequency / returns.count()) - 1
        return compounded
    else:
        return returns.mean() * frequency


def ema_historical_return(
    prices: Union[pd.DataFrame, np.ndarray],
    returns_data: bool = False,
    compounding: bool = True,
    span: int = 500,
    frequency: int = DEFAULT_FREQUENCY
) -> pd.Series:
    """
    Annualized exponentially weighted mean return.

    :param prices: Prices or returns data
    :param returns_data: If True, 'prices' is actually returns
    :param compounding: If True, compute compounded annual return
    :param span: Span for the EMA
    :param frequency: Periods per year
    :return: Series of annual returns
    """
    data = _ensure_dataframe(prices)
    returns = data if returns_data else returns_from_prices(data)
    ewm_mean = returns.ewm(span=span).mean().iloc[-1]

    if compounding:
        return (1 + ewm_mean) ** frequency - 1
    else:
        return ewm_mean * frequency


def james_stein_shrinkage(*args, **kwargs):
    """
    Deprecated: James-Stein shrinkage is no longer supported.
    """
    raise NotImplementedError(
        "James-Stein shrinkage has been removed."
    )


def capm_return(
    prices: Union[pd.DataFrame, np.ndarray],
    market_prices: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    returns_data: bool = False,
    risk_free_rate: float = 0.02,
    compounding: bool = True,
    frequency: int = DEFAULT_FREQUENCY
) -> pd.Series:
    """
    Compute annualized CAPM returns: R = Rf + beta * (Rm - Rf)

    :param prices: Asset prices or returns
    :param market_prices: Market benchmark prices or returns
    :param returns_data: If True, inputs are returns
    :param risk_free_rate: Annual risk-free rate
    :param compounding: If True, compound returns
    :param frequency: Periods per year
    :return: Series of CAPM returns
    """
    asset_df = _ensure_dataframe(prices)
    if not returns_data:
        asset_ret = returns_from_prices(asset_df)
    else:
        asset_ret = asset_df.copy()

    if market_prices is not None:
        mkt_df = _ensure_dataframe(market_prices)
        market_ret = (
            mkt_df if returns_data
            else returns_from_prices(mkt_df)
        )
        market_ret = market_ret.rename(columns={market_ret.columns[0]: 'mkt'})
        asset_ret = asset_ret.join(market_ret, how='left')
    else:
        asset_ret['mkt'] = asset_ret.mean(axis=1)

    cov = asset_ret.cov()
    betas = cov.loc[asset_ret.columns.difference(['mkt']), 'mkt'] \
            / cov.at['mkt', 'mkt']

    if compounding:
        mkt_mean = (1 + asset_ret['mkt']).prod() ** (frequency / asset_ret['mkt'].count()) - 1
    else:
        mkt_mean = asset_ret['mkt'].mean() * frequency

    return pd.Series(risk_free_rate + betas * (mkt_mean - risk_free_rate), name='capm')
