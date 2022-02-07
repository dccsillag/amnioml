"""
This script runs Napari, visualizing our predictions and corresponding exams and ground truths.

Example usage:

python src/eval/models/visualize_predictions.py -p <PATH_TO_HDF_WITH_PREDICTIONS> # visualize all subjects
python src/eval/models/visualize_predictions.py -p <PATH_TO_HDF_WITH_PREDICTIONS> -s <SUBJECT_ID> # visualize a single subject
"""

import argparse

import h5py
import napari
import nrrd
import numpy as np
from vispy.color import Colormap

from src.utils.data_utils import get_nrrd_scale, get_pool_paths_from_subject_id
from src.utils.general_utils import normalize_image

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    "-p",
    default="",
    help="path to hdf with predictions",
)

parser.add_argument(
    "--subjects",
    "-s",
    nargs="*",
    help="which subjects to visualize. If unspecified, then shows all",
)

parser.add_argument(
    "-ns",
    "--no_scaling",
    const=True,
    action="store_const",
    help="disable scaling based on header data",
)
args = parser.parse_args()
f = h5py.File(args.path, "r")

label_blue = Colormap([[0.0, 0.0, 0.0, 0.0], [0, 0, 1, 0.5]])
label_red = Colormap([[0.0, 0.0, 0.0, 0.0], [1, 0, 0, 0.5]])

if args.subjects is None:
    subjects = f.keys()
else:
    subjects = args.subjects

with napari.gui_qt():
    viewer = napari.Viewer()

    for subject in subjects:
        visible = False

        exam_path, segmentation_path = get_pool_paths_from_subject_id(subject)
        exam = nrrd.read(exam_path)
        segmentation = nrrd.read(segmentation_path)
        prediction = np.array(f[subject])
        prediction = np.array((prediction > 0.5), dtype=int)

        exam_scale = get_nrrd_scale(exam[1])
        pred_scale = exam_scale * exam[1]["sizes"] / prediction.shape

        viewer.add_image(
            normalize_image(exam[0]),
            name=f"E_{subject}",
            visible=visible,
            scale=[1, 1, 1] if args.no_scaling else exam_scale,
        )
        viewer.add_image(
            normalize_image(segmentation[0]),
            name=f"S_{subject}",
            visible=visible,
            contrast_limits=[0, 1],
            blending="additive",
            colormap=("truth", label_blue),
            scale=[1, 1, 1] if args.no_scaling else exam_scale,
        )
        viewer.add_image(
            normalize_image(prediction),
            name=f"P_{subject}",
            visible=visible,
            contrast_limits=[0, 1],
            blending="additive",
            colormap=("pred", label_red),
            scale=[1, 1, 1] if args.no_scaling else pred_scale,
        )
