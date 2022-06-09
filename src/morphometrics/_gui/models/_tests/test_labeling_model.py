import numpy as np
from napari.layers import Image, Labels

from morphometrics._gui.models.labeling_model import LabelingModel


def test_labeling_model():
    labeling_model = LabelingModel()
    assert labeling_model.labels_layer is None
    assert labeling_model.background_mask_layer is None
    assert labeling_model.curating is False

    labels_layer = Labels(np.random.randint(0, 5, (10, 10, 10)))
    labeling_model.labels_layer = labels_layer
    assert labeling_model.labels_layer is labels_layer

    background_mask_layer = Image(np.random.randint(0, 1, (10, 10, 10)).astype(bool))
    labeling_model.background_mask_layer = background_mask_layer
    assert labeling_model.background_mask_layer is background_mask_layer

    labeling_model.curating = True
    assert labeling_model.curating is True
