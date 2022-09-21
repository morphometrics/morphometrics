import glob
import os

import h5py
import numpy as np
import tifffile
import wget
import yaml
from plantseg.pipeline.raw2seg import raw2seg
from stardist.models import StarDist3D


def load_to_disk(preprocessed_image):
    """
    If args.model == plant-seg,
    create a directory and write the preprocessed image in it, for raw2seg() to call (raw2seg() requires to load images from disk).

    Parameters
    ----------
    preprocessed_image: np.ndarray
        the preprocessed image

    Returns
    -------
    preprocessed_path: string
        path to the preprocessed images on the disk ('./preprocessed')
    """

    parent_dir = os.getcwd()
    directory = "preprocessed"
    preprocessed_path = os.path.join(parent_dir, directory)
    if not os.path.exists(preprocessed_path):
        os.mkdir(preprocessed_path)

    tifffile.imwrite(f"{preprocessed_path}/temp.tif", preprocessed_image)

    return preprocessed_path


def make_config(preprocessed_image):
    """
    If args.model == plant-seg,
    make the plantseg default yaml or local (with file name "config.yaml") configuration.

    Parameters
    -----------
    preprocessed_image: np.ndarray
        the preprocessed image

    Returns
    -----------
    config: Python object corresponding to the defined configuration
        the user-defined configuration
    """

    file = r".\config.yaml"
    if not os.path.exists(file):
        # if no local configuration, load the plant-seg example yaml file from github
        url = "https://raw.githubusercontent.com/hci-unihd/plant-seg/master/examples/config.yaml"
        file = wget.download(url)
    config = yaml.load(open(file), Loader=yaml.FullLoader)
    config["path"] = load_to_disk(preprocessed_image)

    return config


def segment(model, preprocessed_image):
    """
    Perform segmentation using different models.

    Parameters
    -----------
    model: string
        the model to use for segmentation. Options: "plant-seg", "stardist"
    preprocessed_image: np.ndarray
        the preprocessed image

    Returns
    -----------
    segmented_image: np.ndarray
        the segmented image

    """
    if model == "plant-seg":
        # do plant-seg segmentation
        config = make_config(preprocessed_image)
        raw2seg(config)  # outputs to disk
        # load from disk
        # this is the path using default plantseg parameters, if made changeble should ajust here as well
        path = os.path.join(
            os.getcwd(),
            "preprocessed\\PreProcessing\\generic_confocal_3d_unet\\MultiCut",
        )
        files = glob.iglob(f"{path}/*.h5")
        latest = max(files, key=os.path.getctime)
        hf = h5py.File(latest, "r")
        segmented_image = np.array(
            hf["segmentation"][:]
        )  # why is the name 'segmentation'?

    if model == "stardist":
        # do stardist segmentation
        model = StarDist3D.from_pretrained("3D_demo")
        segmented_image, details = model.predict_instances(preprocessed_image)

    return segmented_image
