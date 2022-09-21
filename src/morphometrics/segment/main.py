import argparse
import os
from timeit import default_timer as timer

import tifffile

from ..segment import post_process, pre_process, segment


def parser():
    """
    Define the parameters for the segmentation pipeline.

    Returns
    -------
    args: object
        user-defined parameters
    """

    parser = argparse.ArgumentParser(
        description="3d cell image segmentation using plant-seg pipeline"
    )
    parser.add_argument(
        "--datadir",
        required=True,
        help="Path to input directory containing images to process.",
    )
    parser.add_argument(
        "--maskdir",
        required=True,
        help="Path to input directory containing image masks correspondingly.",
    )
    parser.add_argument(
        "--raw_pixel_size", required=True, help="Raw pixel size of the image."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="The model used for segmentation. Availabel options: plant-seg, stardist.",
    )
    parser.add_argument(
        "--target_pixel_size",
        required=True,
        help="Target pixel size of the input images of the selected algorithm.",
    )
    parser.add_argument(
        "--threshold",
        required=True,
        help="Size threshold of the cell in the number of pixels, any cell smaller than the threshold will be deleted.",
    )

    args = parser.parse_args()

    return args


def main():
    args = parser()

    # all_images = glob.iglob(f'{args.datadir}/*')
    all_images = os.listdir(args.datadir)
    all_masks = os.listdir(args.maskdir)
    if not os.path.exists("./postprocessed"):
        os.mkdir("./postprocessed")

    for image, mask in zip(all_images, all_masks):
        timer1 = timer()
        raw_image = tifffile.imread(os.path.join(args.datadir, image))
        timer2 = timer()
        print(f"raw data read successful, cost {timer2 - timer1}s")

        # process 1 image
        segmentation_mask = tifffile.imread(os.path.join(args.maskdir, mask))
        preprocessed_image = pre_process.pre_process_image(
            raw_image, segmentation_mask, args.raw_pixel_size, args.target_pixel_size
        )
        timer3 = timer()
        print(f"preprocessing successful, cost {timer3 - timer2}s")

        segmented_image = segment.segment(args.model, preprocessed_image)
        timer4 = timer()
        print(f"segmentation successful, cost {timer4 - timer3}s")

        postprocessed_image = post_process.post_process_image(
            segmented_image, segmentation_mask, args.threshold
        )
        timer5 = timer()
        print(f"postprocessing successful, cost {timer5 - timer4}s")

        # save
        tifffile.imwrite(
            f"./postprocessed/postprocessed_{image.strip('')}", postprocessed_image
        )
        print("pipeline finished!")


if __name__ == "__main__":
    main()
