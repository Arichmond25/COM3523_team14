''' 
plotting.py

Module for plotting portfolio analytics: covariance/correlation matrices, dendrograms,
efficient frontiers, and weight distributions.
'''
import logging
from typing import Optional, Union, Dict, Tuple

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from matplotlib import pyplot as plt

from .risk_models import cov_to_corr

# Module constants
DEFAULT_DPI: int = 300
DEFAULT_SHOW: bool = True
DEFAULT_STYLE: str = 'seaborn-deep'

logger = logging.getLogger(__name__)


def _save_and_show(
    ax, filename: Optional[str] = None, dpi: int = DEFAULT_DPI, show: bool = DEFAULT_SHOW
) -> None:
    '''
    Finalize plot: apply style, save to file, and display.
    '''
    plt.style.use(DEFAULT_STYLE)
    fig = ax.get_figure()
    fig.tight_layout()
    try:
        if filename:
            fig.savefig(fname=filename, dpi=dpi)
    except Exception as e:
        logger.error("Failed to save figure '%s': %s", filename, e)
    if show:
        plt.show()


def plot_covariance(
    cov_matrix: Union[pd.DataFrame, np.ndarray],
    plot_correlation: bool = False,
    show_tickers: bool = True,
    filename: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
    show: bool = DEFAULT_SHOW
) -> plt.Axes:
    '''
    Plot a covariance or correlation matrix as a heatmap.
    '''
    if isinstance(cov_matrix, np.ndarray):
        matrix = pd.DataFrame(cov_matrix)
    else:
        matrix = cov_matrix.copy()
    if plot_correlation:
        matrix = cov_to_corr(matrix)

    fig, ax = plt.subplots()
    cax = ax.imshow(matrix.values)
    fig.colorbar(cax, ax=ax)

    if show_tickers and hasattr(matrix, 'index'):
        ticks = np.arange(matrix.shape[0])
        ax.set_xticks(ticks)
        ax.set_xticklabels(matrix.index, rotation=90)
        ax.set_yticks(ticks)
        ax.set_yticklabels(matrix.index)

    _save_and_show(ax, filename, dpi, show)
    return ax


def plot_dendrogram(
    hrp,  # HRPOpt instance
    show_tickers: bool = True,
    filename: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
    show: bool = DEFAULT_SHOW
) -> plt.Axes:
    '''
    Plot a hierarchical clustering dendrogram from an HRPOpt object.
    '''
    if not hasattr(hrp, 'clusters') or hrp.clusters is None:
        hrp.optimize()

    fig, ax = plt.subplots()
    params = {'labels': hrp.tickers} if show_tickers else {'no_labels': True}
    sch.dendrogram(hrp.clusters, ax=ax, orientation='top', **params)
    if show_tickers:
        plt.xticks(rotation=90)

    _save_and_show(ax, filename, dpi, show)
    return ax


def plot_efficient_frontier(
    cla,  # CLA or EfficientFrontier instance
    points: int = 100,
    show_assets: bool = True,
    filename: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
    show: bool = DEFAULT_SHOW
) -> plt.Axes:
    '''
    Plot the efficient frontier, marking the optimal portfolio and optionally assets.
    '''
    if getattr(cla, 'frontier_values', None) is None:
        cla.efficient_frontier(points=points)
    mus, sigmas, _ = cla.frontier_values

    fig, ax = plt.subplots()
    ax.plot(sigmas, mus, label='Efficient frontier')

    if show_assets and hasattr(cla, 'cov_matrix') and hasattr(cla, 'expected_returns'):
        risks = np.sqrt(np.diag(cla.cov_matrix))
        ax.scatter(risks, cla.expected_returns, s=30, label='assets')

    opt_ret, opt_vol, _ = cla.portfolio_performance()
    ax.scatter(opt_vol, opt_ret, marker='x', s=100, label='optimal')
    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    ax.legend()

    _save_and_show(ax, filename, dpi, show)
    return ax


def plot_weights(
    weights: Union[Dict[str, float], pd.Series],
    filename: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
    show: bool = DEFAULT_SHOW
) -> plt.Axes:
    '''
    Plot portfolio weights as a horizontal bar chart.
    '''
    if isinstance(weights, dict):
        series = pd.Series(weights)
    else:
        series = weights.sort_values(ascending=False)

    fig, ax = plt.subplots()
    series.plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel('Weight')

    _save_and_show(ax, filename, dpi, show)
    return ax