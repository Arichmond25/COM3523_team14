"""
The ``hierarchical_portfolio`` module seeks to implement one of the recent advances in
portfolio optimisation – the application of hierarchical clustering models in allocation.

All of the hierarchical classes have a similar API to ``EfficientFrontier``, though since
many hierarchical models currently don't support different objectives, the actual allocation
happens with a call to `optimize()`.
"""

import collections
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

from . import base_optimizer, risk_models


class HRPOpt(base_optimizer.BaseOptimizer):
    """
    A HRPOpt object constructs a Hierarchical Risk Parity portfolio.

    Instance variables:
    - ``returns``: pd.DataFrame of asset returns (or None if using a precomputed covariance).
    - ``cov_matrix``: pd.DataFrame of asset covariances (or None if using returns).
    - ``clusters``: linkage matrix from the last `optimize()` call.

    Public methods:
    - `optimize()`: computes and sets `self.weights` (an OrderedDict).
    - `portfolio_performance()`: returns (return, vol, Sharpe) for the last weights.
    """

    def __init__(self, returns=None, cov_matrix=None):
        if returns is None and cov_matrix is None:
            raise ValueError("Either returns or cov_matrix must be provided")

        if returns is not None:
            if not isinstance(returns, pd.DataFrame):
                raise TypeError("returns are not a dataframe")
            tickers = list(returns.columns)
        else:
            tickers = list(cov_matrix.columns)

        self.returns = returns
        self.cov_matrix = cov_matrix
        self.clusters = None
        super().__init__(len(tickers), tickers)

    @staticmethod
    def _get_cluster_var(cov, cluster_items):
        """Compute the variance of a cluster via inverse-variance weighting."""
        cov_slice = cov.loc[cluster_items, cluster_items]
        ivp = 1 / np.diag(cov_slice)
        ivp /= ivp.sum()
        return np.linalg.multi_dot((ivp, cov_slice, ivp))

    @staticmethod
    def _get_quasi_diag(link):
        """Traverse the linkage tree in pre-order to get a sorted list of leaf indices."""
        return sch.to_tree(link, rd=False).pre_order()

    @staticmethod
    def _raw_hrp_allocation(cov, ordered_tickers):
        """Recursively allocate weights by bisecting clusters and applying HRP."""
        w = pd.Series(1, index=ordered_tickers)
        clusters = [ordered_tickers]
        while clusters:
            # bisect each cluster
            clusters = [
                group[j:k]
                for group in clusters
                for j, k in ((0, len(group)//2), (len(group)//2, len(group)))
                if len(group) > 1
            ]
            # allocate within each pair
            for i in range(0, len(clusters), 2):
                left, right = clusters[i], clusters[i+1]
                vl = HRPOpt._get_cluster_var(cov, left)
                vr = HRPOpt._get_cluster_var(cov, right)
                α = 1 - vl / (vl + vr)
                w[left]   *= α
                w[right] *= (1 - α)
        return w

    def _get_corr_and_cov(self):
        """
        Return (corr, cov) depending on whether `self.returns` or
        `self.cov_matrix` was provided.
        """
        if self.returns is None:
            cov  = self.cov_matrix
            corr = risk_models.cov_to_corr(cov).round(6)
        else:
            corr = self.returns.corr()
            cov  = self.returns.cov()
        return corr, cov

    @staticmethod
    def _compute_distance(corr):
        """Convert a correlation matrix into a SciPy distance vector."""
        m = np.sqrt(np.clip((1.0 - corr) / 2.0, a_min=0.0, a_max=1.0))
        return ssd.squareform(m, checks=False)

    def optimize(self, linkage_method="single"):
        """
        Build the HRP portfolio:
        1. get corr & cov
        2. form distance matrix
        3. cluster via scipy.linkage
        4. sort leaves via pre-order
        5. allocate via `_raw_hrp_allocation`
        """
        if linkage_method not in sch._LINKAGE_METHODS:
            raise ValueError("linkage_method must be one recognised by scipy")

        corr, cov = self._get_corr_and_cov()
        dist      = HRPOpt._compute_distance(corr)
        self.clusters = sch.linkage(dist, linkage_method)

        order = corr.index[HRPOpt._get_quasi_diag(self.clusters)].tolist()
        raw_w = HRPOpt._raw_hrp_allocation(cov, order)
        weights = collections.OrderedDict(raw_w.sort_index())

        self.set_weights(weights)
        return weights

    def portfolio_performance(self, verbose=False, risk_free_rate=0.02, frequency=252):
        """
        After `optimize()`, compute (ret, vol, Sharpe). Raises ValueError if
        called before `optimize()`.
        """
        if self.weights is None:
            raise ValueError("Weights have not been calculated yet")

        if self.returns is None:
            mu  = None
            cov = self.cov_matrix
        else:
            mu  = self.returns.mean() * frequency
            cov = self.returns.cov()  * frequency

        return base_optimizer.portfolio_performance(
            self.weights, mu, cov, verbose, risk_free_rate
        )
