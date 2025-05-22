''' 
hierarchical_portfolio.py

Implements Hierarchical Risk Parity (HRP) portfolio allocation.

Classes:
    HRPOpt: Hierarchical Risk Parity optimizer inheriting from BaseOptimizer.
'''  
import logging
from collections import OrderedDict
from typing import Optional, List, Union, Tuple

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

from .base_optimizer import BaseOptimizer
from .risk_models import cov_to_corr

# Module constants
DEFAULT_LINKAGE_METHOD = 'single'
DEFAULT_RISK_FREE_RATE = 0.02
DEFAULT_FREQUENCY = 252

logger = logging.getLogger(__name__)


def _ensure_dataframe(
    data: Union[pd.DataFrame, np.ndarray]
) -> pd.DataFrame:
    """
    Ensure data is a pandas DataFrame. Convert from ndarray if needed.
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()
    try:
        return pd.DataFrame(data)
    except Exception as e:
        logger.error("Could not convert data to DataFrame: %s", e)
        raise ValueError("Input must be a pandas DataFrame or convertible ndarray.")


def _compute_distance_matrix(
    corr: pd.DataFrame
) -> np.ndarray:
    """
    Compute the distance matrix from a correlation matrix for clustering.
    """
    # distance = sqrt((1 - corr) / 2)
    clipped = np.clip((1.0 - corr.values) / 2.0, 0.0, 1.0)
    dist_vec = np.sqrt(clipped[np.triu_indices_from(clipped, k=1)])
    return ssd.squareform(dist_vec, checks=False)


class HRPOpt(BaseOptimizer):
    """
    Hierarchical Risk Parity optimizer.
    """
    def __init__(
        self,
        returns: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        cov_matrix: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ):
        if returns is None and cov_matrix is None:
            raise ValueError("Either returns or cov_matrix must be provided.")
        if returns is not None:
            returns_df = _ensure_dataframe(returns)
        else:
            returns_df = None
        if cov_matrix is not None:
            cov_df = _ensure_dataframe(cov_matrix)
        else:
            cov_df = None

        tickers = (
            list(returns_df.columns)
            if returns_df is not None else
            list(cov_df.columns)
        )
        super().__init__(n_assets=len(tickers), tickers=tickers)

        self.returns: Optional[pd.DataFrame] = returns_df
        self.cov_matrix: Optional[pd.DataFrame] = cov_df
        self.clusters: Optional[np.ndarray] = None

    @staticmethod
    def _get_cluster_var(
        cov: pd.DataFrame,
        cluster: List[str]
    ) -> float:
        """
        Compute inverse-variance cluster variance.
        """
        slice_df = cov.loc[cluster, cluster]
        inv_var = 1.0 / np.diag(slice_df)
        weights = inv_var / inv_var.sum()
        return float(weights @ slice_df.values @ weights)

    @staticmethod
    def _get_quasi_diag(
        linkage: np.ndarray
    ) -> List[int]:
        """
        Traverse linkage matrix to get order of tickers.
        """
        tree = sch.to_tree(linkage, rd=False)
        return tree.pre_order()

    @classmethod
    def _allocate_weights(
        cls,
        cov: pd.DataFrame,
        ordered: List[str]
    ) -> pd.Series:
        """
        Compute HRP weights by recursive bisection.
        """
        weights = pd.Series(1.0, index=ordered)
        clusters = [ordered]
        # Recursively split clusters
        while clusters:
            next_clusters = []
            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                mid = len(cluster) // 2
                left, right = cluster[:mid], cluster[mid:]
                var_left = cls._get_cluster_var(cov, left)
                var_right = cls._get_cluster_var(cov, right)
                alpha = 1 - var_left / (var_left + var_right)
                weights[left] *= alpha
                weights[right] *= (1 - alpha)
                next_clusters.extend([left, right])
            clusters = next_clusters
        return weights

    def optimize(
        self,
        linkage_method: str = DEFAULT_LINKAGE_METHOD
    ) -> OrderedDict:
        """
        Perform HRP allocation, returning weights.
        """
        if linkage_method not in sch._LINKAGE_METHODS:
            raise ValueError(f"Unknown linkage method '{linkage_method}'.")

        # Compute covariance and correlation
        if self.cov_matrix is not None:
            cov = self.cov_matrix
        else:
            cov = _ensure_dataframe(self.returns).cov()

        corr = cov_to_corr(cov).round(6)
        dist = _compute_distance_matrix(corr)
        linkage = sch.linkage(dist, method=linkage_method)

        ordered_idx = self._get_quasi_diag(linkage)
        ordered = list(corr.index[ordered_idx])

        raw_weights = self._allocate_weights(cov, ordered)
        sorted_weights = raw_weights.sort_index()

        self.clusters = linkage
        self.set_weights(OrderedDict(sorted_weights.to_dict()))
        return self.weights

    def portfolio_performance(
        self,
        verbose: bool = False,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        frequency: int = DEFAULT_FREQUENCY
    ) -> Tuple[float, float, float]:
        """
        Calculate expected return, vol, Sharpe ratio after optimize().
        """
        if not hasattr(self, 'weights') or self.weights is None:
            raise ValueError("Weights have not been set. Call optimize() first.")

        if self.returns is not None:
            rets = self.returns
            mu = rets.mean() * frequency
            cov = rets.cov() * frequency
        else:
            mu = None
            cov = self.cov_matrix * frequency

        return super().portfolio_performance(
            self.weights, mu, cov, verbose, risk_free_rate
        )
