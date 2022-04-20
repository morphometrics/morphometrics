from morphometrics.explore._tests._explore_test_utils import make_test_features_anndata
from morphometrics.explore.cluster import cluster_features
from morphometrics.explore.explore_utils import compute_neighbor_graph


def test_leiden_clustering_no_neighbor_no_gpu():
    """Test leden clustering with pre-computed neighbor graph

    This test doesn't check correctness of the clustering, just that it
    runs and adds the correct fields
    """
    adata = make_test_features_anndata()
    compute_neighbor_graph(adata, use_gpu=False)

    cluster_features(adata, method="leiden", compute_neighbors=False)
    assert "leiden" in adata.obs.columns


def test_leiden_clustering_no_gpu():
    """Test leiden clustering and calculate nighbor graph in the function.

    This test doesn't check correctness of the clustering, just that it
    runs and adds the correct fields
    """
    adata = make_test_features_anndata()

    cluster_features(adata, method="leiden", compute_neighbors=True)
    assert "leiden" in adata.obs.columns
