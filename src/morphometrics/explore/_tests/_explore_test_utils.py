import anndata
import numpy as np


def make_test_features_anndata() -> anndata.AnnData:
    rng = np.random.default_rng(42)
    X = rng.random((50, 10))

    return anndata.AnnData(X=X)
