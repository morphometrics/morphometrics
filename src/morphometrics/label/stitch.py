from typing import Dict, List, Optional, Tuple

import numpy as np
from skimage.measure import regionprops_table
from tqdm.auto import tqdm

from morphometrics.types import LabelImage


def find_segment_centroid(
    label_image: LabelImage, label_value: int, z_index: Optional[int] = None
) -> np.ndarray:

    ndim = label_image.ndim
    if ndim == 2:
        indices = np.argwhere(label_image == label_value)
    elif ndim == 3:
        indices = np.argwhere(label_image[z_index, :, :] == label_value)
    else:
        raise ValueError("labeL_image must be 2D or 3D")

    return np.mean(indices, axis=0).astype(int)


def _measure_segment_properties(
    label_image: LabelImage, axis: int = 0, background_label: int = 0
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """Measure the thickness of all segments along a given axis.

    Parameters
    ----------
    label_image : LabelImage
        The labels to compute the thickness on.
    axis : int
        The axis along which to compute thickness

    """
    region_props = regionprops_table(
        label_image=label_image, properties=("label", "bbox")
    )

    # get the keys for the bbox measurements
    bbox_min_label = axis
    bbox_max_label = label_image.ndim + axis

    # get the bbox_arrays
    bbox_min = region_props[f"bbox-{bbox_min_label}"]
    bbox_max = region_props[f"bbox-{bbox_max_label}"]

    # compute the segment thicknesses
    thicknesses = bbox_max - bbox_min
    segment_thicknesses = {
        label_value: thickness
        for label_value, thickness in zip(region_props["label"], thicknesses)
        if label_value != background_label
    }

    # compute the coordinates - bbox is a half-open interval
    all_top_coordinates = bbox_max - 1
    top_coordinates = {
        label_value: coordinate
        for label_value, coordinate in zip(region_props["label"], all_top_coordinates)
        if label_value != background_label
    }
    bottom_coordinates = {
        label_value: coordinate
        for label_value, coordinate in zip(region_props["label"], bbox_min)
        if label_value != background_label
    }

    return segment_thicknesses, top_coordinates, bottom_coordinates


def _find_labels_to_stitch(
    label_image: LabelImage,
    starting_label: int,
    starting_coordinate: int,
    coordinate_increment: int,
    segment_thicknesses: Dict[int, int],
    labels_to_stitch: Optional[List[int]] = None,
    thickness_threshold: int = 7,
    center_distance_threshold: float = 50,
    background_label: Optional[int] = 0,
) -> List[int]:

    if labels_to_stitch is None:
        # default to empty list if no indices are provided
        labels_to_stitch = []

    # iteration cannot exceed the shape of the image
    maximum_coordinate = label_image.shape[0] - 1
    minimum_coordinate = 0

    # initialize the state variables
    new_coordinate = starting_coordinate
    current_label = starting_label
    previous_centroid = find_segment_centroid(
        label_image, starting_label, starting_coordinate
    )

    end_reached = False

    while not end_reached:

        # if np.any(np.isnan(previous_centroid)):
        #     break

        new_label = label_image[
            int(new_coordinate), previous_centroid[0], previous_centroid[1]
        ]

        if (background_label is not None) and (new_label == background_label):
            # stop iterating if a background label was reached
            break

        if new_label != current_label:
            # if entering a new label, check if it should be stitched
            previous_coordinate = new_coordinate - coordinate_increment

            # check that the center of the interfacing labels are close
            new_centroid = find_segment_centroid(label_image, new_label, new_coordinate)
            # previous_centroid = find_segment_centroid(label_image, current_label, previous_coordinate)
            center_to_center_distance = np.linalg.norm(new_centroid - previous_centroid)
            centers_close = center_to_center_distance - center_distance_threshold

            # check that the area of the new label is smaller than the previous
            previous_area = np.count_nonzero(
                label_image[previous_coordinate, :, :] == current_label
            )
            new_area = np.count_nonzero(label_image[new_coordinate, :, :] == new_label)
            area_decreasing = new_area <= previous_area

            # check the new segments thickness is below the threshold
            thickness = segment_thicknesses[new_label]
            thickness_valid = thickness < thickness_threshold

            if centers_close and area_decreasing and thickness_valid:
                # if stitching criteria met, add label to list and continue
                labels_to_stitch.append(new_label)
                current_label = new_label

            else:
                # a tip is reached, stop iterating
                end_reached = True
                break

        # update the centroid
        previous_centroid = find_segment_centroid(
            label_image, current_label, new_coordinate
        )

        # update the coordinate
        new_coordinate += coordinate_increment

        # check if the end image has been reached
        if (new_coordinate > maximum_coordinate) or (
            new_coordinate < minimum_coordinate
        ):
            end_reached = True

    return labels_to_stitch


def find_labels_to_stitch(
    label_image: LabelImage,
    starting_label: int,
    top_coordinates: Dict[int, int],
    bottom_coordinates: Dict[int, int],
    segment_thicknesses: Dict[int, int],
    thickness_threshold: int = 7,
    center_distance_threshold: float = 50,
    background_label: Optional[int] = 0,
) -> List[int]:
    """
    Finds the bad labels that need to be fused with good_label as well as the durations
    taken to find them

    Lines 161-177 :
    Lines 186-214 : finds the centroid and area for the new label's slice and then finds
                    booleans for the validity criteria for the bad labels

    The while loop stops executing when all the bad labels have been found on
    both sides of the good segment

    """

    """ Initialise everything. Selects the 2d slice to begin with and the direction """

    # find the labels to stitch in the positive direction
    labels_to_stitch = _find_labels_to_stitch(
        label_image=label_image,
        starting_label=starting_label,
        starting_coordinate=top_coordinates[starting_label],
        coordinate_increment=1,
        segment_thicknesses=segment_thicknesses,
        labels_to_stitch=[],
        thickness_threshold=thickness_threshold,
        center_distance_threshold=center_distance_threshold,
        background_label=background_label,
    )

    # find the labels to stitch in the negative direction
    labels_to_stitch = _find_labels_to_stitch(
        label_image=label_image,
        starting_label=starting_label,
        starting_coordinate=bottom_coordinates[starting_label],
        coordinate_increment=-1,
        segment_thicknesses=segment_thicknesses,
        labels_to_stitch=labels_to_stitch,
        thickness_threshold=thickness_threshold,
    )

    return labels_to_stitch


def stitch_from_selected_labels(
    label_image: LabelImage,
    starting_labels: List[int],
    thickness_threshold: int = 7,
    center_distance_threshold: float = 50,
    background_label: Optional[int] = 0,
    in_place: bool = False,
    verbose: bool = False,
) -> np.ndarray:

    if not in_place:
        label_image = label_image.copy()

    (
        segment_thicknesses,
        top_coordinates,
        bottom_coordinates,
    ) = _measure_segment_properties(
        label_image=label_image, axis=0, background_label=background_label
    )

    starting_labels_set = set(starting_labels)
    if verbose:
        starting_labels = tqdm(starting_labels)
    for label_value in starting_labels:
        labels_to_stitch = find_labels_to_stitch(
            label_image=label_image,
            starting_label=label_value,
            top_coordinates=top_coordinates,
            bottom_coordinates=bottom_coordinates,
            segment_thicknesses=segment_thicknesses,
            thickness_threshold=thickness_threshold,
            center_distance_threshold=center_distance_threshold,
            background_label=background_label,
        )

        if len(label_image) == 0:
            # go to the next iteration if there aren't labels to stitch
            continue

        # remove any of the starting labels from the labels to stitch if present
        clean_labels_to_stitch = set(labels_to_stitch).difference(starting_labels_set)

        # update the values to be stitched
        stitching_mask = np.isin(label_image, list(clean_labels_to_stitch))
        label_image[stitching_mask] = label_value

    return label_image
