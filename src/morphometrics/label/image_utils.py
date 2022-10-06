from typing import List, Optional, Tuple, Union

import numpy as np
import pyclesperanto_prototype as cle
from scipy.sparse import csr_matrix

from ..types import BinaryImage, LabelImage


def expand_selected_labels(
    label_image: LabelImage,
    label_values_to_expand: Union[int, List[int]],
    expansion_amount: int = 1,
    background_mask: Optional[np.ndarray] = None,
) -> LabelImage:
    """Expand selected label values in a label image.

    Parameters
    ----------
    label_image : LabelImage
        The label image to expand selected values in. Must be 2D or 3D.
    label_values_to_expand : Union[int, List[int]]
        The values in the label image that are allowed to expand.
    expansion_amount : int
        The radius of the expansion. Expansion is performed with equal
        amounts in all directions.
    background_mask : Optional[np.ndarray]
        A boolean mask where True denotes background into which the labels
        cannot be expanded. If None, the background mask is not applied.
        Default value is None

    Returns
    -------
    expanded_label_image : LabelImage
        The resulting label image with the selected labels expanded.
    """

    if isinstance(label_values_to_expand, int):
        # coerce into list
        label_values_to_expand = [label_values_to_expand]

    if label_image.ndim == 2:
        label_image_is_2d = True
        label_image = np.expand_dims(label_image, axis=0)
    elif label_image.ndim == 3:
        label_image_is_2d = False
    else:
        raise ValueError("label_image must be 2D or 3D")

    gpu_labels = cle.push(label_image)

    # preallocate the memory
    flip = cle.create_labels_like(gpu_labels)
    flop = cle.create_labels_like(gpu_labels)
    flap = cle.create_labels_like(gpu_labels)
    labels_mask = cle.create_labels_like(gpu_labels)
    expanded_labels = cle.create_labels_like(gpu_labels)

    image_changed_flag = cle.create([1, 1, 1])

    # create a mask where the pixels the are allowed to expand are set to True
    expand_mask = cle.create_binary_like(gpu_labels)
    cle.equal_constant(gpu_labels, expand_mask, 0)

    # create a label image containing only the labels that are allowed to expand
    for label_value in label_values_to_expand:
        cle.equal_constant(gpu_labels, labels_mask, label_value)
        cle.binary_or(expand_mask, labels_mask, expand_mask)

    # create the labels to expand
    cle.multiply_images(gpu_labels, expand_mask, flip)

    # perform the expansion
    for i in range(expansion_amount):
        # dilate the labels
        cle.onlyzero_overwrite_maximum_box(flip, image_changed_flag, flop)
        cle.onlyzero_overwrite_maximum_diamond(flop, image_changed_flag, flap)

        # remove the expansion that occurred where other labels already are
        cle.multiply_images(flap, expand_mask, flip)

    # insert the expanded labels back into the original image
    cle.binary_not(expand_mask, labels_mask)
    cle.multiply_images(gpu_labels, labels_mask, expanded_labels)
    cle.add_images(expanded_labels, flip, expanded_labels)

    expanded_label_image = np.asarray(expanded_labels)
    if label_image_is_2d is True:
        expanded_label_image = np.squeeze(expanded_label_image)

    # trim off the expansion that went outside of the tissue
    if background_mask is not None:
        expanded_label_image[background_mask] = 0

    return expanded_label_image


def expand_selected_labels_using_crop(
    label_image: LabelImage,
    label_values_to_expand: Union[int, List[int]],
    expansion_amount: int = 1,
    background_mask: Optional[np.ndarray] = None,
) -> LabelImage:
    """Expand selected label values in a label image by cropping out the bounding box
    surrounding the selected labels and performing the expansion on the crop.

    Parameters
    ----------
    label_image : LabelImage
        The label image to expand selected values in. Must be 2D or 3D.
    label_values_to_expand : Union[int, List[int]]
        The values in the label image that are allowed to expand.
    expansion_amount : int
        The radius of the expansion. Expansion is performed with equal
        amounts in all directions.
    background_mask : Optional[np.ndarray]
        A boolean mask where True denotes background into which the labels
        cannot be expanded. If None, the background mask is not applied.
        Default value is None

    Returns
    -------
    expanded_label_image : LabelImage
        The resulting label image with the selected labels expanded.
    """
    if isinstance(label_values_to_expand, int):
        # coerce int to list
        label_values_to_expand = [label_values_to_expand]
    label_mask = np.zeros_like(label_image, dtype=bool)
    for label_value in label_values_to_expand:
        label_mask = np.logical_or(label_mask, label_image == label_value)

    if expansion_amount > 1:
        bounding_box_expansion = 2 * expansion_amount
    else:
        # when expansion is 1, actual radius ends up being 3
        bounding_box_expansion = 4
    bounding_box = get_mask_bounding_box_3d(label_mask)
    expanded_bounding_box = expand_bounding_box(
        bounding_box=bounding_box,
        expansion_amount=bounding_box_expansion,
        image_shape=label_image.shape,
    )

    # extract a crop around the bounding box of the labels
    expansion_crop = label_image[
        expanded_bounding_box[0, 0] : expanded_bounding_box[0, 1],
        expanded_bounding_box[1, 0] : expanded_bounding_box[1, 1],
        expanded_bounding_box[2, 0] : expanded_bounding_box[2, 1],
    ]

    if background_mask is not None:
        # crop the background mask
        background_mask_crop = background_mask[
            expanded_bounding_box[0, 0] : expanded_bounding_box[0, 1],
            expanded_bounding_box[1, 0] : expanded_bounding_box[1, 1],
            expanded_bounding_box[2, 0] : expanded_bounding_box[2, 1],
        ]
    else:
        background_mask_crop = None

    expanded_crop = expand_selected_labels(
        label_image=expansion_crop,
        label_values_to_expand=label_values_to_expand,
        expansion_amount=expansion_amount,
        background_mask=background_mask_crop,
    )

    # insert the expanded labels back into the original image
    expanded_labels = label_image.copy()
    expanded_labels[
        expanded_bounding_box[0, 0] : expanded_bounding_box[0, 1],
        expanded_bounding_box[1, 0] : expanded_bounding_box[1, 1],
        expanded_bounding_box[2, 0] : expanded_bounding_box[2, 1],
    ] = expanded_crop

    return expanded_labels


def get_mask_bounding_box_3d(mask_image: BinaryImage) -> np.ndarray:
    """Get the axis-aligned bounding box around the True values in a 3D mask.

    Parameters
    ----------
    mask_image : BinaryImage
        The binary image from which to calculate the bounding box.

    Returns
    -------
    bounding_box : np.ndarray
        The bounding box as an array where arranged:
            [
                [0_min, _max],
                [1_min, 1_max],
                [2_min, 2_max]
            ]
        where 0, 1, and 2 are the 0th, 1st, and 2nd dimensions,
        respectively.
    """
    z = np.any(mask_image, axis=(1, 2))
    y = np.any(mask_image, axis=(0, 2))
    x = np.any(mask_image, axis=(0, 1))

    z_min, z_max = np.where(z)[0][[0, -1]]
    y_min, y_max = np.where(y)[0][[0, -1]]
    x_min, x_max = np.where(x)[0][[0, -1]]

    return np.array([[z_min, z_max], [y_min, y_max], [x_min, x_max]], dtype=int)


def expand_bounding_box(
    bounding_box: np.ndarray,
    expansion_amount: int,
    image_shape: Optional[Tuple[int]] = None,
) -> np.ndarray:
    """Expand a bounding box bidirectionally along each axis by a specified amount.

    Parameters
    ----------
    bounding_box : np.ndarray
        The bounding box as an array where arranged:
            [
                [0_min, _max],
                [1_min, 1_max],
                [2_min, 2_max]
            ]
        where 0, 1, and 2 are the 0th, 1st, and 2nd dimensions,
        respectively.
    expansion_amount : int
        The number of pixels to expand the bounding box by in each direction
    image_shape : Tuple[int]
        The size of the image along each axis.

    Returns
    -------
    expanded_bounding_box : np.ndarray
        The expanded bounding box.
    """
    expanded_bounding_box = bounding_box.copy()
    expanded_bounding_box[:, 0] = expanded_bounding_box[:, 0] - expansion_amount
    expanded_bounding_box[:, 1] = expanded_bounding_box[:, 1] + expansion_amount

    if image_shape is not None:
        # max index is image_shape - 1
        max_value = np.asarray(image_shape).reshape((3, 1)) - 1
    else:
        max_value = None

    return np.clip(expanded_bounding_box, a_min=0, a_max=max_value)


def _lower_triangle_to_full_array(lower_triangle_array: np.ndarray) -> np.ndarray:
    """Convert a 2D lower triangle array into a full array.

    Parameters
    ----------
    lower_triangle_array : np.ndarray
        The lower triangle array to fill in.

    Returns
    -------
    full_array : np.ndarray
        The filled in array. Has the same dimensions as input array.
    """
    # ensure lower triangle
    lower_triangle_array = np.tril(lower_triangle_array)

    # create the array by flipping the lower half
    full_array = lower_triangle_array.T + lower_triangle_array
    idx = np.arange(lower_triangle_array.shape[0])

    # add in the diagonal elements
    full_array[idx, idx] = lower_triangle_array[idx, idx]
    return full_array


def touch_matrix_from_label_image(
    label_image: LabelImage,
) -> Tuple[csr_matrix, np.ndarray]:
    """Create a touch matrix from a label image.

    The touch matrix is only between labeled components. Background is assumed to be 0.

    Parameters
    ----------
    label_image : LabelImage
        The label image to generate the touch matrix from.

    Returns
    -------
    touch_matrix : csr_matrix
        The touch matrix in a sparse array.
    touches_background_mask : np.ndarray
        Boolean mask
    """
    # get the touch matrix and expand to a full matrix
    lower_triangle_touch_matrix = np.asarray(cle.generate_touch_matrix(label_image))
    full_touch_matrix = _lower_triangle_to_full_array(lower_triangle_touch_matrix)

    # get the touch matrix without the background touches
    touch_matrix = csr_matrix(full_touch_matrix[1:, 1:])
    touches_background_mask = full_touch_matrix[1:, 0].astype(bool)

    return touch_matrix, touches_background_mask
