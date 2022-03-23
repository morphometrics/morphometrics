import pytest

from morphometrics.explore._tests._explore_test_utils import make_test_features_anndata
from morphometrics.explore.dimensionality_reduction import pca, umap


@pytest.mark.parametrize("normalize_data", [True, False])
def test_pca_no_gpu(normalize_data: bool):
    """This test doesn't check correctness of the PCA in, just that it
    runs and adds the correct fields
    """
    adata = make_test_features_anndata()
    pca(adata, normalize_data=normalize_data)

    assert "X_pca" in adata.obsm_keys()


@pytest.mark.parametrize("normalize_data", [True, False])
def test_umap_no_gpu(normalize_data):
    adata = make_test_features_anndata()
    umap(adata, normalize_data=normalize_data, compute_neighbors=True)

    assert "X_umap" in adata.obsm_keys()
