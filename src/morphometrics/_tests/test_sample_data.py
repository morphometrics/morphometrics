from morphometrics._sample_data import make_random_3d_image, make_simple_labeled_cube


def test_random_3d_image():
    layer_data_list = make_random_3d_image()
    assert len(layer_data_list) == 1

    data, meta, layer_type = layer_data_list[0]
    assert data.ndim == 3
    assert layer_type == "image"


def test_simple_labeled_cube():
    layer_data_list = make_simple_labeled_cube()
    assert len(layer_data_list) == 1

    data, meta, layer_type = layer_data_list[0]
    assert data.ndim == 3
    assert data.dtype == int
    assert layer_type == "labels"
