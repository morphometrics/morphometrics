from typing import Any, Dict, Optional

import anndata
import scanpy as sc

from .explore_utils import compute_neighbor_graph


def pca(
    adata: anndata.AnnData,
    normalize_data: bool = True,
    compute_neighbors: bool = True,
    neighbors_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Optional[anndata.AnnData]:
    """Perform principle component analysis on pre-computer features.

    This is largly a passthrough to the scanpy pca function. PCA coordinates
    are added to adata.obsm["X_pca"]. PCA is only performed on the features
    in adata.X.
    https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.pca.html

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the features to perform PCA on..
    normalize_data : bool
        Flag set to True to normalize data to unit variance and zero mean
        using the scanpy.pp.scale function. Default value is True.
        https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.scale.html
    **kwargs
        Keyword arguments to pass to the scanpy pca function.
        See the scanpy docs for details:
        https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.pca.html

    Returns
    -------
    adata : Optional[anndata.AnnData]
        If copy=True was passed, a new AnnData object is returned with
        the PCA coordinates.
    """
    if normalize_data is True:
        sc.pp.scale(adata)

    return sc.tl.pca(adata, **kwargs)


def umap(
    adata: anndata.AnnData,
    normalize_data: bool = True,
    compute_neighbors: bool = False,
    neighbors_kwargs: Optional[Dict[str, Any]] = None,
    umap_kwargs: Optional[Dict[str, Any]] = None,
    use_gpu: bool = False,
):
    """Perform dimensionality reduction with umap.

    This is largly a passthrough to the scanpy umap function. umap coordinates
    are added to adata.obsm["X_umap"]. umap is only performed on the features
    in adata.X.
    https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.umap.html

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the features to perform umap on.
    normalize_data : bool
        Flag set to True to normalize data to unit variance and zero mean
        using the scanpy.pp.scale function. Default value is True.
        https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.scale.html
    compute_neighbors : bool
        Flag to compute the neighbor graph if set to True. The neighbor graph
        is required for UMAP. This is computed using scanpy.pp.neighbors.
        Default value is True.
    neighbors_kwargs : Optional[Dict[str, Any]]
        Dictionary of keyword arguments to pass to the scanpy neighbors function
        for computing the neighbor graph. See the scanpy docs for details
        https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.neighbors.html
    umap_kwargs : Optional[Dict[str, Any]]
        Dictionary of keyword arguments to pass to the scanpy umap function.
        See the scanpy docs for details:
        https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.umap.html
    use_gpu : bool
        Flag set to True to use the RAPIDS GPU implementation to accelerate
        graph construction. This is equivalent to setting method="rapids"
        in scanpy.pp.neighbors and scanpy.tl.umap.
    **kwargs
        Keyword arguments to pass to the scanpy umap function.
        See the scanpy docs for details

    Returns
    -------
    adata : Optional[anndata.AnnData]
        If copy=True was passed, a new AnnData object is returned with
        the umap coordinates.
    """
    if normalize_data is True:
        sc.pp.scale(adata)

    if use_gpu is True:
        try:
            import cuml  # noqa F401
        except ImportError:
            use_gpu = False

    # neighbor graph
    if compute_neighbors is True:
        if neighbors_kwargs is None:
            neighbors_kwargs = dict()
        compute_neighbor_graph(adata, **neighbors_kwargs)

    # calculate umap
    if umap_kwargs is None:
        umap_kwargs = dict()
    if use_gpu is True:
        umap_kwargs["method"] = "rapids"
    return sc.tl.umap(adata, **umap_kwargs)
