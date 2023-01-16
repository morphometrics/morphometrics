import numpy as np
import pytest

from morphometrics.label.stitch import (
    _find_labels_to_stitch,
    _measure_segment_properties,
    find_labels_to_stitch,
    find_segment_centroid,
    stitch_from_selected_labels,
)


def test_find_segment_centroid_invalid_image():
    """find_segment_centroid() requires 2D or 3D images"""
    with pytest.raises(ValueError):
        _ = find_segment_centroid(
            label_image=np.ones((10, 10, 10, 10)),
            label_value=1,
        )


def test_find_segment_centroid():
    # make a label image
    label_image = np.zeros((10, 10, 10), dtype=int)
    label_image[2:5, 2:5, 5:8] = 1

    centroid = find_segment_centroid(label_image=label_image, label_value=1, z_index=3)
    np.testing.assert_allclose([3, 6], centroid)


def test_find_labels_to_stitch_stop_on_area_increase():
    """The label finding algorithm should stop iterating when a segment with
    increasing area is found.
    """
    # make a label image
    label_image = np.zeros((20, 10, 10), dtype=int)
    label_image[10, 3:8, 3:8] = 1

    # add a tapering profile going up
    label_image[11, 3:7, 3:7] = 2
    label_image[12, 3:7, 3:7] = 3
    label_image[13, 3:8, 3:8] = 4  # tip should be found where area increases

    segment_thicknesses = {1: 1, 2: 1, 3: 1, 4: 1}

    labels_to_stitch = _find_labels_to_stitch(
        label_image=label_image,
        starting_label=1,
        starting_coordinate=10,
        coordinate_increment=1,
        segment_thicknesses=segment_thicknesses,
        labels_to_stitch=[],
        thickness_threshold=5,
    )

    assert set(labels_to_stitch) == {2, 3}


def test_find_labels_to_stitch_stop_on_too_thick():
    """The label finding algorithm should stop iterating when a segment with
    a thickness greater than that threshold is found.
    """
    # make a label image
    label_image = np.zeros((20, 10, 10), dtype=int)
    label_image[10, 3:8, 3:8] = 1

    # add a tapering profile going up
    label_image[11, 3:7, 3:7] = 2
    label_image[12, 3:7, 3:7] = 3
    label_image[13:19, 3:6, 3:6] = 4  # tip should be found where segment is too thick

    thickness_threshold = 5
    segment_thicknesses = {
        1: 1,
        2: 1,
        3: 1,
        4: thickness_threshold + 1,  # tip should be found where segment is too thick
    }

    labels_to_stitch = _find_labels_to_stitch(
        label_image=label_image,
        starting_label=1,
        starting_coordinate=10,
        coordinate_increment=1,
        segment_thicknesses=segment_thicknesses,
        labels_to_stitch=[],
        thickness_threshold=thickness_threshold,
    )
    assert set(labels_to_stitch) == {2, 3}


def test_find_labels_to_stitch_stop_image_boundary():
    """The label finding algorithm should stop iterating when it reaches the
    end of the image.
    """
    # make a label image
    label_image = np.zeros((14, 10, 10), dtype=int)
    label_image[10, 3:8, 3:8] = 1

    # add a tapering profile going up
    label_image[11, 3:7, 3:7] = 2
    label_image[12, 3:7, 3:7] = 3
    label_image[13, 3:6, 3:6] = 4  # tip should be found where segment is too thick

    thickness_threshold = 5
    segment_thicknesses = {
        1: 1,
        2: 1,
        3: 1,
        4: 1,  # tip should be found where segment is too thick
    }

    labels_to_stitch = _find_labels_to_stitch(
        label_image=label_image,
        starting_label=1,
        starting_coordinate=10,
        coordinate_increment=1,
        segment_thicknesses=segment_thicknesses,
        labels_to_stitch=[],
        thickness_threshold=thickness_threshold,
    )
    assert set(labels_to_stitch) == {2, 3, 4}


def test_find_labels_negative_increment():
    """The label finding algorithm should be able to iterate in the negative direction."""
    # make a label image
    label_image = np.zeros((14, 10, 10), dtype=int)
    label_image[10, 3:8, 3:8] = 1

    # add a tapering profile going up
    label_image[9, 3:7, 3:7] = 2
    label_image[8, 3:7, 3:7] = 3
    label_image[7, 3:8, 3:8] = 4  # tip should be found where segment is too thick

    thickness_threshold = 5
    segment_thicknesses = {
        1: 1,
        2: 1,
        3: 1,
        4: 1,  # tip should be found where segment is too thick
    }

    labels_to_stitch = _find_labels_to_stitch(
        label_image=label_image,
        starting_label=1,
        starting_coordinate=10,
        coordinate_increment=-1,
        segment_thicknesses=segment_thicknesses,
        labels_to_stitch=[],
        thickness_threshold=thickness_threshold,
    )
    assert set(labels_to_stitch) == {2, 3}


def test_measure_segment_properties():
    # make a label image
    label_image = np.zeros((20, 10, 10), dtype=int)
    label_image[10, 3:8, 3:8] = 1

    # add a tapering profile going up
    label_image[11, 3:7, 3:7] = 2
    label_image[12, 3:7, 3:7] = 3
    label_image[13:19, 3:6, 3:6] = 4

    background_label = 0
    (
        segment_thicknesses,
        top_coordinates,
        bottom_coordinates,
    ) = _measure_segment_properties(
        label_image=label_image, axis=0, background_label=background_label
    )

    # check the thicknesses
    expected_segment_thicknesses = {1: 1, 2: 1, 3: 1, 4: 6}
    assert background_label not in segment_thicknesses
    assert segment_thicknesses == expected_segment_thicknesses

    # check the top coordinates
    expected_top_coordiantes = {1: 10, 2: 11, 3: 12, 4: 18}
    assert background_label not in top_coordinates
    assert top_coordinates == expected_top_coordiantes

    # check the bottom coordinates
    expected_bottom_coordinates = {1: 10, 2: 11, 3: 12, 4: 13}
    assert background_label not in bottom_coordinates
    assert bottom_coordinates == expected_bottom_coordinates


def test_find_labels_to_stitch():
    # make a label image
    label_image = np.zeros((20, 10, 10), dtype=int)
    label_image[10, 3:8, 3:8] = 1

    # add a tapering profile going up
    label_image[11, 3:7, 3:7] = 2
    label_image[12, 3:7, 3:7] = 3
    label_image[13:15, 3:6, 3:6] = 4
    label_image[15, 3:8, 3:8] = 5  # iteration should stop when segment gets bigger

    # add a tapering profile going down
    label_image[9, 3:7, 3:7] = 6
    label_image[8, 3:7, 3:7] = 7
    label_image[0:7, 3:8, 3:8] = 8  # tip should be found where segment is too thick

    background_label = 0
    (
        segment_thicknesses,
        top_coordinates,
        bottom_coordinates,
    ) = _measure_segment_properties(
        label_image=label_image, axis=0, background_label=background_label
    )

    labels_to_stitch = find_labels_to_stitch(
        label_image=label_image,
        starting_label=1,
        top_coordinates=top_coordinates,
        bottom_coordinates=bottom_coordinates,
        segment_thicknesses=segment_thicknesses,
        thickness_threshold=5,
        background_label=background_label,
    )
    assert set(labels_to_stitch) == {2, 3, 4, 6, 7}


@pytest.mark.parametrize(
    "in_place,verbose", [(True, True), (False, True), (True, False), (False, False)]
)
def test_stitch_from_selected_labels(in_place, verbose):
    # make a label image
    label_image = np.zeros((20, 10, 10), dtype=int)
    label_image[10, 3:8, 3:8] = 1

    # add a tapering profile going up
    label_image[11, 3:7, 3:7] = 2
    label_image[12, 3:7, 3:7] = 3
    label_image[13:15, 3:6, 3:6] = 4
    label_image[15, 3:8, 3:8] = 5  # iteration should stop when segment gets bigger

    # add a tapering profile going down
    label_image[9, 3:7, 3:7] = 6
    label_image[8, 3:7, 3:7] = 7
    label_image[0:7, 3:8, 3:8] = 8  # tip should be found where segment is too thick

    background_label = 0

    stitched_image = stitch_from_selected_labels(
        label_image=label_image,
        starting_labels=[1],
        thickness_threshold=5,
        background_label=background_label,
        in_place=in_place,
        verbose=verbose,
    )

    assert (stitched_image is label_image) is in_place

    # the stitched label values should no longer be present
    expected_stitched_labels = [2, 3, 4, 6, 7]
    assert np.sum(np.isin(stitched_image, expected_stitched_labels)) == 0

    expected_remaining_labels = [0, 1, 5, 8]
    for label_value in expected_remaining_labels:
        assert np.sum(np.isin(stitched_image, [label_value])) > 0
