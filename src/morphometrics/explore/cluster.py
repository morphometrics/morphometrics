from enum import Enum
from typing import Any, Dict, Optional, Union

import anndata
import scanpy as sc

from .explore_utils import compute_neighbor_graph


class ClusteringMethod(Enum):
    LEIDEN = "leiden"


_CLUSTERING_FUNCTIONS = {ClusteringMethod.LEIDEN: sc.tl.leiden}


def cluster_features(
    adata: anndata.AnnData,
    method: Union[str, ClusteringMethod] = "leiden",
    compute_neighbors: bool = False,
    neighbors_kwargs: Optional[Dict[str, Any]] = None,
    clustering_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Optional[anndata.AnnData]:
    """Perform clustering on pre-computer features.

    Cluster identities are added to an obs column with the same name as the clustering algorithm.
    For example, if leiden clustering is used, clusters are accessible adata.obs["leiden"]

    This is largely a passthrough to scanpy clustering methods.
    Available clustering methods:
        leiden: https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.leiden.html

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the features to cluster.
    method : Union[str, ClusteringMethod]
        The clustering method to use. Valid methods are: "leiden"
    compute_neighbors : bool
        Flag to compute the neighbor graph if set to True. The neighbor graph
        is required for clustering. This is computed using scanpy.pp.neighbors.
        Default value is True.
    neighbors_kwargs : Optional[Dict[str, Any]]
        Dictionary of keyword arguments to pass to the scanpy neighbors function
        for computing the neighbor graph. See the scanpy docs for details
        https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.neighbors.html
    clustering_kwargs : Optional[Dict[str, Any]]
        Dictionary of keyword arguments to pass to the scanpy umap function.
        See the scanpy docs for the selected clustering function for details

    Returns
    -------
    adata : Optional[anndata.AnnData]
        If copy=True was passed, a new AnnData object is returned with
        the clusters added.
    """
    if not isinstance(method, ClusteringMethod):
        method = ClusteringMethod(method)
    clustering_function = _CLUSTERING_FUNCTIONS[method]

    # neighbor graph
    if compute_neighbors is True:
        if neighbors_kwargs is None:
            neighbors_kwargs = dict()
        compute_neighbor_graph(adata, **neighbors_kwargs)
    if clustering_kwargs is None:
        clustering_kwargs = dict()
    return clustering_function(adata, **clustering_kwargs)
