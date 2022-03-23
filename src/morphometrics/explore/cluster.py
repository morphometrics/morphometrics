from enum import Enum
from typing import Optional, Union

import anndata
import scanpy as sc


class ClusteringMethod(Enum):
    LEIDEN = "leiden"


_CLUSTERING_FUNCTIONS = {ClusteringMethod.LEIDEN: sc.tl.leiden}


def cluster_features(
    adata: anndata.AnnData, method: Union[str, ClusteringMethod] = "leiden", **kwargs
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
    **kwargs
        Keyword arguments to pass to the clustering algorithm.
        See the scanpy docs for details

    Returns
    -------
    adata : Optional[anndata.AnnData]
        If copy=True was passed, a new AnnData object is returned with
        the clusters added.
    """
    if not isinstance(method, ClusteringMethod):
        method = ClusteringMethod(method)
    clustering_function = _CLUSTERING_FUNCTIONS[method]

    return clustering_function(adata, **kwargs)
