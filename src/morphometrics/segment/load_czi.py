def load_czi(image_file, channel):
    """
    Prepare the 3d czi image to pre-process. For now only load the czi file, if necessary may add downsampling with denoising and partition
    due to the expensive computation of the large file.

    Parameters
    --------
    image_file: path
        path to the image to prepare.
    channel: int
        the channel to load and analyze.

    Returns
    -------
    raw_image: np.ndarray
        raw image for preprocessing
    """
    from aicsimageio import AICSImage

    # available_channels = ["DAPI", "Alexa488", "Alexa568", "Alexa647"]
    # if channel not in available_channels:
    #     raise ValueError(
    #         f"Not an available channel, must be one of {available_channels}."
    #     )

    image = AICSImage(image_file)
    raw_image = image.data[0, channel, :, :, :]

    return raw_image
