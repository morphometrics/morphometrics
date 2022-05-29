from typing import List, Optional, Union

import numpy as np
import pyclesperanto_prototype as cle

from ..types import LabelImage


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

    # insert the expanded labels back into the original immage
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
