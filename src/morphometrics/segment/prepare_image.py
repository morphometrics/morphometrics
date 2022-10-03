def prepare_image(image_file):
    """
    Prepare the 3d czi image to pre-process. For now only load the czi file, if necessary may add downsampling with denoising and partition
    due to the expensive computation of the large file.

    Parameters
    --------
    image_file: path
        path to the image to prepare.

    Returns
    -------
    raw_image: np.ndarray
        raw image for preprocessing
    """
    from aicsimageio import AICSImage

    image = AICSImage(image_file)
    raw_image = image.data[0, 0, :, :, :]  # assuming the first channel is the nuclei

    return raw_image
