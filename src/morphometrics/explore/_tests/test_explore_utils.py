from morphometrics.explore._tests._explore_test_utils import make_test_features_anndata
from morphometrics.explore.explore_utils import compute_neighbor_graph


def test_neighbors_no_gpu():
    """This test doesn't check accuracy of the neighbor graph, just that it
    runs and adds the correct fields"""
    adata = make_test_features_anndata()
    compute_neighbor_graph(adata, use_gpu=False)

    assert "neighbors" in adata.uns
    assert "connectivities" in adata.obsp.keys()
