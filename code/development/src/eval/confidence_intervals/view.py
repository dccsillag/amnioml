import os
from argparse import ArgumentParser

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from medpy.io import load
from tqdm import tqdm
from vispy.color import ColorArray

from src.utils.data_utils import get_pool_paths_from_subject_id
from src.utils.general_utils import outpath

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--prediction_path", help="Path to the predictions HDF of a model")
    parser.add_argument("-H", "--hdf_path")
    args = parser.parse_args()

    assert os.path.dirname(args.hdf_path) == os.path.dirname(args.prediction_path)
    output_directory = os.path.join(os.path.dirname(args.hdf_path), "confidence_interval_imgs-bates_et_al")

    LOWER_SET_RGB = tuple(ColorArray((100, 0.85, 0.9), color_space="hsv").rgb[0, :])
    UPPER_SET_RGB = tuple(ColorArray((0, 0.85, 0.9), color_space="hsv").rgb[0, :])
    PREDICTION_RGB = (0, 0, 0)

    with h5py.File(args.hdf_path, "r") as confsets_file:
        with h5py.File(args.prediction_path, "r") as preds_file:
            upper_sets = confsets_file["upper_sets"]
            lower_sets = confsets_file["lower_sets"]
            subject_ids = list(upper_sets.keys())
            assert list(lower_sets.keys()) == subject_ids

            for subject_id in tqdm(subject_ids, desc="subject ids"):
                segmentation, _ = load(get_pool_paths_from_subject_id(subject_id)[1])
                segmentation = (segmentation > 0.5).astype(np.float64)
                upper_set = upper_sets[subject_id]
                lower_set = lower_sets[subject_id]
                prediction = preds_file[subject_id]

                selected_slice = segmentation.shape[0] // 2
                segmentation = segmentation[selected_slice, :, :]
                upper_set = upper_set[selected_slice, :, :]
                lower_set = lower_set[selected_slice, :, :]
                prediction = prediction[selected_slice, :, :]

                fig, (ax0, ax1) = plt.subplots(1, 2)
                ax0.set_ylabel("prediciton & confidence sets")
                ax0.imshow(upper_set, cmap=ListedColormap([(0, 0, 0, 0), UPPER_SET_RGB]), alpha=0.6)
                ax0.imshow(prediction, cmap=ListedColormap([(0, 0, 0, 0), PREDICTION_RGB]))
                ax0.imshow(lower_set, cmap=ListedColormap([(0, 0, 0, 0), LOWER_SET_RGB]), alpha=0.6)
                ax1.set_ylabel("segmentation")
                ax1.imshow(segmentation, cmap=ListedColormap([(0, 0, 0, 0), (0, 0, 0, 1)]), alpha=0.45)

                fig.savefig(outpath(os.path.join(output_directory, f"{subject_id}.png")), dpi=500)
