import argparse
import os
from timeit import default_timer as timer

import tifffile

from morphometrics.segment.load_czi import load_czi
from morphometrics.segment.post_process import post_process_image
from morphometrics.segment.pre_process import pre_process_image
from morphometrics.segment.segment import segment


def segmentation_parser():
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
        nargs="?",
        help="Path to input directory containing image masks correspondingly, \
                                filename sequences must be the same as image name sequences.\
                                For plant-seg pipeline, it is required while for stardist it is not.",
    )
    parser.add_argument(
        "--raw_pixel_size",
        nargs="*",
        type=float,
        required=True,
        help="Raw pixel size of the image. No comma should be included in the input. Example input: 0.5 0.5 0.5",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="The model used for segmentation. Availabel options: plant-seg, stardist.",
    )
    parser.add_argument(
        "--channel",
        nargs="*",
        type=int,
        help='Specify which channel (int) to segment upon.',
    )
    parser.add_argument(
        "--target_pixel_size",
        nargs="*",
        type=float,
        required=True,
        help="Target pixel size of the input images for the selected pretrained model. No comma should be included in the input. Example input: 0.5 0.5 0.5",
    )
    parser.add_argument(
        "--n_tiles",
        nargs="*",
        type=int,
        help="When using stardist, n_tile is the number of tiles to use for prediction.Must be an iterable of length 3 (the number of dimensions).\
                Example input: 20 20 20",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        required=True,
        help="Size threshold of the cell in the number of pixels, any cell smaller than the threshold will be deleted.",
    )
    args = parser.parse_args()

    return args


def main():
    args = segmentation_parser()

    # all_images = glob.iglob(f'{args.datadir}/*')
    all_images = os.listdir(args.datadir)

    if not os.path.exists("./postprocessed"):
        os.mkdir("./postprocessed")

    if args.model == "plant-seg":
        all_masks = os.listdir(args.maskdir)

        for image, mask in zip(all_images, all_masks):
            start = timer()
            raw_image = tifffile.imread(os.path.join(args.datadir, image))
            read_finish_time = timer()
            print(f"raw data read successful, cost {read_finish_time - start}s")

            # process 1 image
            segmentation_mask = tifffile.imread(os.path.join(args.maskdir, mask))
            preprocessed_image = pre_process_image(
                raw_image,
                args.raw_pixel_size,
                args.target_pixel_size,
                segmentation_mask,
            )
            preprocess_finish_time = timer()
            print(
                f"preprocessing successful, cost {preprocess_finish_time - read_finish_time}s"
            )

            segmented_image = segment(args.model, preprocessed_image)
            segment_finish_time = timer()
            print(
                f"segmentation successful, cost {segment_finish_time - preprocess_finish_time}s"
            )

            postprocessed_image = post_process_image(
                segmented_image, args.threshold, segmentation_mask
            )
            postprocess_finish_time = timer()
            print(
                f"postprocessing successful, cost {postprocess_finish_time - segment_finish_time}s"
            )

            # save
            tifffile.imwrite(
                f"./postprocessed/postprocessed_{image.strip('')}", postprocessed_image
            )
            print("pipeline finished!")

    if args.model == "stardist":
        for image in all_images:
            start = timer()
            raw_image = load_czi(os.path.join(args.datadir, image), args.channel)
            read_finish_time = timer()
            print(f"raw data read successful, cost {read_finish_time - start}s")

            preprocessed_image = pre_process_image(
                raw_image, args.raw_pixel_size, args.target_pixel_size
            )
            preprocess_finish_time = timer()
            print(
                f"preprocessing successful, cost {preprocess_finish_time - read_finish_time}s"
            )

            segmented_image = segment(args.model, preprocessed_image, args.n_tiles)
            segment_finish_time = timer()
            print(
                f"segmentation successful, cost {segment_finish_time - preprocess_finish_time}s"
            )

            postprocessed_image = post_process_image(
                segmented_image, args.threshold, None, raw_image.shape
            )
            postprocess_finish_time = timer()
            print(
                f"postprocessing successful, cost {postprocess_finish_time - segment_finish_time}s"
            )

            # save
            name = image.strip("").replace("czi", "tif")
            tifffile.imwrite(
                f"./postprocessed/postprocessed_{name}", postprocessed_image
            )
            print("pipeline finished!")


if __name__ == "__main__":
    main()
