import numpy as np
import pytest

from morphometrics._gui.label_curator.substack_viewer import SubStackViewer


def test_init_substack_viewer(make_napari_viewer):
    """Test initializing the substack viewer"""
    viewer = make_napari_viewer()
    substack_viewer = SubStackViewer(viewer=viewer)

    assert substack_viewer.parameters_set is False


def test_substack_viewer_3d(make_napari_viewer):
    """The viewer must be in 2D rendering mode."""
    viewer = make_napari_viewer()
    substack_viewer = SubStackViewer(viewer=viewer)

    viewer.dims.ndisplay = 3
    with pytest.raises(ValueError):
        _ = substack_viewer.normal_vector


def test_substack_viewer_2d(make_napari_viewer):
    """The image must be 3D"""
    viewer = make_napari_viewer()
    substack_viewer = SubStackViewer(viewer=viewer)
    viewer.add_image(np.zeros((10, 10)))

    with pytest.raises(ValueError):
        _ = substack_viewer.normal_vector


def test_subsample(make_napari_viewer):
    viewer = make_napari_viewer()
    substack_viewer = SubStackViewer(viewer=viewer)

    test_volume = np.zeros((10, 10, 20), dtype=int)
    test_volume[3:7, 3:7, :6] = 1
    test_volume[3:7, 3:7, 6:15] = 2
    test_volume[3:7, 3:7, 15:] = 1
    labels_layer = viewer.add_labels(test_volume)

    half_width = 1
    substack_viewer.set_sample_parameters(layer=labels_layer, half_width=half_width)
    substack_viewer.start_point = np.array([5, 5, 4])
    substack_viewer.end_point = np.array([5, 5, 10])

    assert substack_viewer.half_width == half_width
    assert substack_viewer.layer is labels_layer

    # slice along the 1 axis
    viewer.dims.order = (1, 0, 2)
    np.testing.assert_equal([0, 1, 0], substack_viewer.normal_vector)

    subvolume = substack_viewer.sample_subvolume_from_line_segment()

    expected_subvolume = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ]
    ).reshape(1, 7, 3, 3)
    np.testing.assert_equal(subvolume, expected_subvolume)

    expected_sampling_coordinates = np.array(
        [
            [6.0, 4.0, 4.0],
            [5.0, 4.0, 4.0],
            [4.0, 4.0, 4.0],
            [6.0, 5.0, 4.0],
            [5.0, 5.0, 4.0],
            [4.0, 5.0, 4.0],
            [6.0, 6.0, 4.0],
            [5.0, 6.0, 4.0],
            [4.0, 6.0, 4.0],
            [6.0, 4.0, 5.0],
            [5.0, 4.0, 5.0],
            [4.0, 4.0, 5.0],
            [6.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
            [4.0, 5.0, 5.0],
            [6.0, 6.0, 5.0],
            [5.0, 6.0, 5.0],
            [4.0, 6.0, 5.0],
            [6.0, 4.0, 6.0],
            [5.0, 4.0, 6.0],
            [4.0, 4.0, 6.0],
            [6.0, 5.0, 6.0],
            [5.0, 5.0, 6.0],
            [4.0, 5.0, 6.0],
            [6.0, 6.0, 6.0],
            [5.0, 6.0, 6.0],
            [4.0, 6.0, 6.0],
            [6.0, 4.0, 7.0],
            [5.0, 4.0, 7.0],
            [4.0, 4.0, 7.0],
            [6.0, 5.0, 7.0],
            [5.0, 5.0, 7.0],
            [4.0, 5.0, 7.0],
            [6.0, 6.0, 7.0],
            [5.0, 6.0, 7.0],
            [4.0, 6.0, 7.0],
            [6.0, 4.0, 8.0],
            [5.0, 4.0, 8.0],
            [4.0, 4.0, 8.0],
            [6.0, 5.0, 8.0],
            [5.0, 5.0, 8.0],
            [4.0, 5.0, 8.0],
            [6.0, 6.0, 8.0],
            [5.0, 6.0, 8.0],
            [4.0, 6.0, 8.0],
            [6.0, 4.0, 9.0],
            [5.0, 4.0, 9.0],
            [4.0, 4.0, 9.0],
            [6.0, 5.0, 9.0],
            [5.0, 5.0, 9.0],
            [4.0, 5.0, 9.0],
            [6.0, 6.0, 9.0],
            [5.0, 6.0, 9.0],
            [4.0, 6.0, 9.0],
            [6.0, 4.0, 10.0],
            [5.0, 4.0, 10.0],
            [4.0, 4.0, 10.0],
            [6.0, 5.0, 10.0],
            [5.0, 5.0, 10.0],
            [4.0, 5.0, 10.0],
            [6.0, 6.0, 10.0],
            [5.0, 6.0, 10.0],
            [4.0, 6.0, 10.0],
        ]
    ).reshape(1, 7, 3, 3, 3)
    np.testing.assert_almost_equal(
        expected_sampling_coordinates, substack_viewer._sampling_coordinates
    )
