"""
Utils for the eval pipeline.
"""

from typing import Dict, Iterator, List, Tuple, Union

import h5py
import numpy as np
from scipy.ndimage import binary_erosion, label
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

from src.utils.data_utils import volume_from_mask


class IterIdExamTargetPred:
    def __init__(self, predictions_hdf_path: str, *, metadata: bool = False):
        self.file = h5py.File(predictions_hdf_path, "r")
        self.subjects = self.file.keys()
        self.metadata = metadata

    def __iter__(
        self,
    ) -> Iterator[
        Union[
            Tuple[str, np.ndarray, np.ndarray, np.ndarray],
            Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ]
    ]:
        for subject in self.subjects:
            subject_group = self.file[subject]
            out = (
                subject,
                np.asarray(subject_group["exam"]),
                np.asarray(subject_group["segmentation"]),
                np.asarray(subject_group["prediction"]),
            )
            if self.metadata:
                yield out + (
                    np.asarray(subject_group["exam_transform"]),
                    np.asarray(subject_group["exam_origin"]),
                    np.asarray(subject_group["segmentation_transform"]),
                    np.asarray(subject_group["segmentation_origin"]),
                )
            else:
                yield out

    def __len__(self) -> int:
        return len(self.subjects)


def mean_squared_error(
    input_grid: np.ndarray, true_segmentation: np.ndarray, predicted_segmentation: np.ndarray
) -> float:
    input_grid = np.asarray(input_grid)
    true_segmentation = np.asarray(true_segmentation)
    predicted_segmentation = np.asarray(predicted_segmentation)

    return np.mean((true_segmentation - predicted_segmentation) ** 2)


SEGMENTATION_METRICS = {"mean_squared_error": mean_squared_error}


def eval_metrics_segmentation(
    input_grid: np.ndarray,
    true_segmentation: np.ndarray,
    predicted_segmentation: np.ndarray,
) -> Dict[str, float]:
    input_grid = np.asarray(input_grid)
    true_segmentation = np.asarray(true_segmentation)
    predicted_segmentation = np.asarray(predicted_segmentation)

    output = {}

    for metric_name, func in SEGMENTATION_METRICS.items():
        metric_value = func(input_grid, true_segmentation, predicted_segmentation)
        output[metric_name] = metric_value

    return output


def get_border(img: np.ndarray) -> np.ndarray:
    eroded = binary_erosion(img)
    return img ^ eroded


def hausdorff_distance(u: np.ndarray, v: np.ndarray) -> float:
    # Adapted from the end of
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html
    u = np.argwhere(get_border(u))
    v = np.argwhere(get_border(v))
    return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])


def dice_coefficient(segmentation: np.ndarray, target: np.ndarray, assertion: bool = False) -> float:
    # Binarize segmentation using a threshold of 0.5.
    # If segmentation is already binary, it will remain the same.
    segmentation = (segmentation >= 0.5) + 0  # sum zero to cast booleans to integers
    if assertion:  # asserts make the function considerably slower
        assert segmentation.shape == target.shape, "Segmentation and target do not have the same size."
        assert set(np.unique(segmentation)).issubset(
            set([0, 1])
        ), "There are values in the segmentation that are neither zero nor one."
        assert set(np.unique(target)).issubset(
            set([0, 1])
        ), "There are values in the target that are neither zero nor one."
    return 2 * (segmentation * target).sum() / (segmentation.sum() + target.sum())


def dice_coefficients_from_hdf5s(segmentations_path: str, targets_path: str) -> List[float]:
    segmentations = h5py.File(segmentations_path, "r")
    targets = h5py.File(targets_path, "r")["segmentations"]
    assert segmentations.keys() == targets.keys(), "Segmentation and target files have different subjects."
    dices = []
    for subject in tqdm(targets.keys()):
        dices.append(
            dice_coefficient(
                np.asarray(segmentations[subject]).transpose(2, 0, 1),
                np.asarray([np.asarray(targets[subject][f"{slice}"]) for slice in range(len(targets[subject]))]),
            )
        )
    return dices


def volumes_from_hdf5s(segmentations_path: str, targets_path: str) -> Tuple[List[float], List[float]]:
    segmentations = h5py.File(segmentations_path, "r")
    targets = h5py.File(targets_path, "r")["segmentations"]
    assert segmentations.keys() == targets.keys(), "Segmentation and target files have different subjects."
    segmentations_volumes = []
    targets_volumes = []
    for subject in tqdm(targets.keys()):
        segmentations_volumes.append(
            volume_from_mask(
                (np.asarray(segmentations[subject]).transpose(2, 0, 1) >= 0.5) + 0,
                subject,
            )
        )
        targets_volumes.append(
            volume_from_mask(
                np.asarray([np.asarray(targets[subject][f"{slice}"]) for slice in range(len(targets[subject]))]),
                subject,
            )
        )
    return (segmentations_volumes, targets_volumes)


def volume(segmentation: np.ndarray, space_directions: np.ndarray) -> float:
    assert set(np.unique(segmentation)).issubset({0, 1})
    volume = (
        np.count_nonzero(segmentation)
        * np.linalg.norm(space_directions[0])
        * np.linalg.norm(space_directions[1])
        * np.linalg.norm(space_directions[2])
    )
    return volume / 1000.0
