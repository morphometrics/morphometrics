from morphometrics.explore._tests._explore_test_utils import make_test_features_anndata
from morphometrics.explore.cluster import cluster_features
from morphometrics.explore.explore_utils import compute_neighbor_graph


def test_leiden_clustering_no_gpu():
    """This test doesn't check correctness of the clusterin, just that it
    runs and adds the correct fields"""
    adata = make_test_features_anndata()
    compute_neighbor_graph(adata, use_gpu=False)

    cluster_features(adata, method="leiden")
    assert "leiden" in adata.obs.columns
