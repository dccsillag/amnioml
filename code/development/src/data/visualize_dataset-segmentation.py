"""
This script runs Napari on a given dataset.

Example usage:

python src/data/visualize_dataset-segmentation.py -d segmentation_700
"""

import argparse
import glob

import napari
import nrrd
import numpy as np
from vispy.color import Colormap

from src.utils.data_utils import get_3d_convex_hull, get_exam_gradient, get_nrrd_scale, get_pool_paths_from_subject_id
from src.utils.general_utils import normalize_image

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    "-d",
    default="segmentation_199",
    help="Name of dataset to use",
)
parser.add_argument(
    "--subjects",
    "-s",
    default=["774451"],
    nargs="+",
    help="Which subject to visualize",
)

parser.add_argument(
    "-ns",
    "--no_scaling",
    const=True,
    action="store_const",
    help="disable scaling based on header data",
)

parser.add_argument(
    "-grad",
    "--gradient",
    const=True,
    action="store_const",
    help="include gradient in visualization",
)

parser.add_argument(
    "-grad-gamma",
    "--gradient-gamma",
    default=0.3,
    help="set gamma for the gradient layers",
)

parser.add_argument(
    "--mask-gradient-with-convex-hull",
    "-maskg",
    const=True,
    action="store_const",
    help="use the convex hull of the segmentation as a mask to the gradient",
)

parser.add_argument(
    "--personal-folder",
    "-pf",
    const=True,
    action="store_const",
    help="use personal folder instead of shared folder",
)

args = parser.parse_args()


exams = []
segmentations = []

if args.dataset == "pool":
    for subject in args.subjects:
        exam_path, segmentation_path = get_pool_paths_from_subject_id(subject)
        exams.append(nrrd.read(exam_path))
        segmentations.append(nrrd.read(segmentation_path))


else:
    exam_filename = "1" if args.dataset == "raw" else "exam"
    segmentations_filename = "2" if args.dataset == "raw" else "segmentation"
    data_root = "personal/data/" if args.personal_folder else "data/"
    subject_folders = [glob.glob(f"{data_root}{args.dataset}/*/{subject}")[0] for subject in args.subjects]
    for subject_folder in subject_folders:
        exams.append(nrrd.read(f"{subject_folder}/{exam_filename}.nrrd"))
        segmentations.append(nrrd.read(f"{subject_folder}/{segmentations_filename}.nrrd"))

red = Colormap([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]])
green = Colormap([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
blue = Colormap([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
label_orange = Colormap([[0.0, 0.0, 0.0, 0.0], [255 / 255, 165 / 255, 0 / 255, 0.8]])
visible = True if len(args.subjects) < 2 else False

with napari.gui_qt():
    viewer = napari.Viewer()
    for subject, exam, segmentation in zip(args.subjects, exams, segmentations):
        viewer.add_image(
            normalize_image(exam[0]),
            name=f"E_{subject}",
            visible=visible,
            scale=[1, 1, 1] if args.no_scaling else get_nrrd_scale(exam[1]),
        )
        viewer.add_image(
            normalize_image(segmentation[0]),
            name=f"S_{subject}",
            visible=visible,
            contrast_limits=[0, 1],
            colormap=("label_orange", label_orange),
            scale=[1, 1, 1] if args.no_scaling else get_nrrd_scale(segmentation[1]),
        )

        if args.gradient:
            grad = np.abs(get_exam_gradient(exam[0], exam[1]))
            if args.mask_gradient_with_convex_hull:
                convex_hull = get_3d_convex_hull(segmentation)
                viewer.add_image(
                    normalize_image(convex_hull),
                    name=f"Convex_hull_{subject}",
                    visible=visible,
                    contrast_limits=[0, 1],
                    colormap=("label_orange", label_orange),
                    scale=[1, 1, 1] if args.no_scaling else get_nrrd_scale(segmentation[1]),
                )

                for i in range(3):
                    grad[i] = np.multiply(convex_hull, grad[i])

            gamma = float(args.gradient_gamma)
            viewer.add_image(
                normalize_image(grad[0]),
                name=f"E_{subject} - gradient0 (red)",
                blending="additive",
                visible=False,
                colormap=("red", red),
                scale=[1, 1, 1] if args.no_scaling else get_nrrd_scale(exam[1]),
                gamma=gamma,
            )
            viewer.add_image(
                normalize_image(grad[1]),
                name=f"E_{subject} - gradient1 (green)",
                blending="additive",
                visible=False,
                colormap=("green", green),
                scale=[1, 1, 1] if args.no_scaling else get_nrrd_scale(exam[1]),
                gamma=gamma,
            )
            viewer.add_image(
                normalize_image(grad[2]),
                name=f"E_{subject} - gradient2 (blue)",
                blending="additive",
                visible=False,
                colormap=("blue", blue),
                scale=[1, 1, 1] if args.no_scaling else get_nrrd_scale(exam[1]),
                gamma=gamma,
            )
