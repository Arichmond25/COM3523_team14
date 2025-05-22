import numpy as np
import scipy.cluster.hierarchy as sch
from . import risk_models

try:
    import matplotlib.pyplot as plt
    plt.style.use("classic")
except (ModuleNotFoundError, ImportError):
    raise ImportError("Please install matplotlib via pip or poetry")


def _plot_io(filename=None, showfig=True, dpi=300):
    """
    Helper to adjust layout, optionally save, and show the figure.
    """
    plt.tight_layout()
    if filename:
        plt.savefig(fname=filename, dpi=dpi)
    if showfig:
        plt.show()


def plot_covariance(
    cov_matrix,
    plot_correlation=False,
    show_tickers=True,
    filename=None,
    showfig=True,
    dpi=300,
):
    """
    Plot the covariance (or correlation) matrix.
    """
    matrix = (
        risk_models.cov_to_corr(cov_matrix)
        if plot_correlation
        else cov_matrix
    )
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix)
    fig.colorbar(cax)

    if show_tickers:
        labels = matrix.index
        ticks = np.arange(len(labels))
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        plt.xticks(rotation=90)

    _plot_io(filename=filename, showfig=showfig, dpi=dpi)
    return ax


def plot_dendrogram(
    hrp,
    show_tickers=True,
    filename=None,
    showfig=True,
    dpi=300,
):
    """
    Plot the hierarchical clustering dendrogram for an HRPOpt instance.
    """
    if hrp.clusters is None:
        hrp.optimize()

    fig, ax = plt.subplots()
    if show_tickers:
        # draw leaves with labels
        sch.dendrogram(
            hrp.clusters,
            labels=hrp.tickers,
            ax=ax,
            orientation="top",
        )
        # match original layout call before rotation
        plt.tight_layout()
        # rotate labels
        plt.xticks(rotation=90)
    else:
        sch.dendrogram(hrp.clusters, no_labels=True, ax=ax)

    # then do our usual save/show and final tight_layout
    _plot_io(filename=filename, showfig=showfig, dpi=dpi)
    return ax


def plot_efficient_frontier(
    cla,
    points=100,
    show_assets=True,
    filename=None,
    showfig=True,
    dpi=300,
):
    """
    Plot the efficient frontier based on a CLA object.
    """
    if cla.weights is None:
        cla.max_sharpe()
    optimal_ret, optimal_risk, _ = cla.portfolio_performance()

    if cla.frontier_values is None:
        cla.efficient_frontier(points=points)

    mus, sigmas, _ = cla.frontier_values
    fig, ax = plt.subplots()
    ax.plot(sigmas, mus, label="Efficient frontier")

    if show_assets:
        ax.scatter(
            np.sqrt(np.diag(cla.cov_matrix)),
            cla.expected_returns,
            s=30,
            color="k",
            label="assets",
        )

    ax.scatter(
        optimal_risk,
        optimal_ret,
        marker="x",
        s=100,
        color="r",
        label="optimal",
    )
    ax.legend()
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")

    _plot_io(filename=filename, showfig=showfig, dpi=dpi)
    return ax


def plot_weights(
    weights,
    filename=None,
    showfig=True,
    dpi=300,
):
    """
    Plot the portfolio weights as a horizontal bar chart.
    """
    # sort descending
    items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    labels, vals = zip(*items)
    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots()
    ax.barh(y_pos, vals)
    ax.set_xlabel("Weight")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    _plot_io(filename=filename, showfig=showfig, dpi=dpi)
    return ax
