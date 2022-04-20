from typing import Optional

import anndata
import scanpy as sc


def compute_neighbor_graph(
    adata: anndata.AnnData, use_gpu: bool = False, **kwargs
) -> Optional[anndata.AnnData]:
    """Create a neighbor graph on pre-computed features

    This is largly a passthrough to the scanpy neighbors function.
    neighbor graph is only constructed with features in adata.X.
    https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.neighbors.html

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing the features to perform umap on.
    use_gpu : bool
        Flag set to True to use the RAPIDS GPU implementation to accelerate
        graph construction. This is equivalent to setting method="rapids"
        in scanpy.pp.neighbors
    **kwargs
        Keyword arguments to pass to the scanpy umap function.
        See the scanpy docs for details

    Returns
    -------
    adata : Optional[anndata.AnnData]
        If copy=True was passed, a new AnnData object is returned with
        the neighbor graph added.
    """
    if use_gpu is True:
        try:
            import cuml  # noqa F401
        except ImportError:
            use_gpu = False

    # neighbor graph
    if use_gpu is True:
        kwargs["method"] = "rapids"
    sc.pp.neighbors(adata, **kwargs)
