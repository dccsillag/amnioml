"""
Utils for the data pipeline.
"""

import glob
import hashlib
import itertools
import os
import shutil
from secrets import token_hex
from tempfile import mkstemp
from typing import Callable, Dict, Iterator, List, Optional, OrderedDict, Tuple, Union

import h5py
import hdf5plugin
import nrrd
import numpy as np
import torch
import torchio as tio
from tqdm import tqdm

PATH_POOL_OF_NRRDS = "data/pool_of_nrrds"
PATH_BLACKLISTED_IDS = "src/data/blacklisted-ids.txt"

EXAM_BASENAMES = set({"1.nrrd", "exam.nrrd"})
SEGMENTATION_BASENAMES = set({"2.nrrd", "segmentation.nrrd"})


def volume_from_mask(data: np.array, subject_id: str) -> float:
    """ Return volume of nonzero elements in mL """

    header = get_header(subject_id)
    scale = get_nrrd_scale(header)
    volume = np.count_nonzero(data) * scale[0] * scale[1] * scale[2]
    return volume / 1000.0


def total_variation(tensor: np.array, scale: np.array) -> float:
    grad = np.array(get_scaled_gradient(tensor, scale))
    abs_grad = np.square(grad[0, :, :]) + np.square(grad[1, :, :]) + np.square(grad[2, :, :])
    return np.sum(np.sqrt(abs_grad))


def total_variation_aniso(tensor: np.array, scale: np.array) -> float:
    grad = np.array(get_scaled_gradient(tensor, [1, 1, 1]))
    abs_grad = np.abs(grad[0, :, :]) + np.abs(grad[1, :, :]) + np.abs(grad[2, :, :])
    return np.sum(abs_grad)


def get_header(subject_id: str) -> OrderedDict:
    """Return the header for subject_id, without reading the entire nrrd file."""

    e_path, s_path = get_pool_paths_from_subject_id(subject_id)
    nrrd_header = nrrd.read_header(e_path)
    return nrrd_header


def get_normalized_exam_slice(exam_slice: np.array) -> np.array:
    """Return the normalized exam slice."""

    multiplier = 1 / max(
        np.max(
            exam_slice[
                int(exam_slice.shape[0] / 4) : int(exam_slice.shape[0] * 3 / 4),
                int(exam_slice.shape[1] / 4) : int(exam_slice.shape[1] * 3 / 4),
            ]
        ),
        1e-4,
    )
    multiplier = multiplier * 0.98

    # the line bellow probably can be optimized
    out = np.minimum(exam_slice * multiplier, np.ones(exam_slice.shape))

    return out


class HdfDataset3d(torch.utils.data.Dataset):
    """Data loader for the hdf_3d format.

    Note: some segmentations have more than 2 values, so normalizing by max or
    ptp would have undesirable results. Hence, the following convention was adopted:

    Normalize the segmentation with a nonzero test.
    """

    def __init__(
        self,
        subject_folders: List[str],
        exam_transforms: List[Callable[[torch.Tensor], torch.Tensor]],
        segmentation_transforms: List[Callable[[torch.Tensor], torch.Tensor]],
    ) -> None:

        self.subject_folders = subject_folders
        self.exam_transforms = exam_transforms
        self.segmentation_transforms = segmentation_transforms
        self.inputs_dtype = torch.float32
        self.outputs_dtype = torch.float32

    def __len__(self) -> int:
        return len(self.subject_folders)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filepath = self.subject_folders[index]
        file = h5py.File(filepath, "r")
        exam = file["exam"][:]
        segmentation = file["segmentation"][:]

        # Normalize to 0-1
        max_exam_inv = 1.0 / max(exam.max(), 1.0)
        exam = np.array(exam, dtype=np.float32) * max_exam_inv
        segmentation = np.array(segmentation != 0, dtype=np.float32)

        # Convert to torch tensor
        exam = torch.from_numpy(exam)
        segmentation = torch.from_numpy(segmentation)

        # Change order to [depth, height, width]
        exam = exam.permute(2, 0, 1)
        segmentation = segmentation.permute(2, 0, 1)

        # Defining augmentations for exams only
        if self.exam_transforms is not None:
            for transform in self.exam_transforms:
                exam = transform(exam)
        if self.segmentation_transforms is not None:
            for transform in self.segmentation_transforms:
                segmentation = transform(segmentation)

        # Typecasting
        exam = exam.to(self.inputs_dtype)
        segmentation = segmentation.to(self.outputs_dtype)

        return exam.unsqueeze(0), segmentation


class VolumeRegressionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        subject_folders: List[str],
        exam_transforms: List[Callable[[torch.Tensor], torch.Tensor]],
        segmentation_transforms: List[Callable[[torch.Tensor], torch.Tensor]],
    ) -> None:
        self.dataset_3d = HdfDataset3d(subject_folders, exam_transforms, segmentation_transforms)
        self.subject_folders = subject_folders

        self.inputs_dtype = torch.float32
        self.outputs_dtype = torch.float32

    def __len__(self) -> int:
        return len(self.dataset_3d)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.subject_folders[index]
        subject_id = os.path.splitext(os.path.split(file_path)[1])[0]
        exam, segmentation = self.dataset_3d[index]
        volume = volume_from_mask(segmentation, subject_id)

        # Typecasting
        exam = exam.to(self.inputs_dtype)
        volume = torch.as_tensor(volume, dtype=self.outputs_dtype)

        return exam, volume


class SegmentationDataset3D(torch.utils.data.Dataset):
    def __init__(
        self,
        subject_folders: List[str],
        transforms: Optional[List[Callable[[torch.tensor], torch.tensor]]] = None,
    ) -> None:

        self.subject_folders = subject_folders
        self.transforms = transforms
        self.inputs_dtype = torch.float32
        self.outputs_dtype = torch.long

    def __len__(self) -> int:
        return len(self.subject_folders)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        folder_path = self.subject_folders[index]
        exam = nrrd.read(os.path.join(folder_path, "exam.nrrd"))[0]
        segmentation = nrrd.read(os.path.join(folder_path, "segmentation.nrrd"))[0]

        # Normalize to 0-1
        exam = (exam - np.min(exam)) / np.ptp(exam)
        segmentation = (segmentation - np.min(segmentation)) / np.ptp(segmentation)

        # Convert to torch tensor
        exam = torch.from_numpy(exam)
        segmentation = torch.from_numpy(segmentation)

        # Change order to [depth, height, width]
        exam = exam.permute(2, 0, 1)
        segmentation = segmentation.permute(2, 0, 1)

        # Defining augmentations for exams only
        transforms_for_exams_only = [random_bias_magnect_field_3d, random_ghosting_3d]

        if self.transforms is not None:
            for transform in self.transforms:
                exam = transform(exam)
                if transform not in transforms_for_exams_only:
                    segmentation = transform(segmentation)
        # Typecasting
        exam = exam.to(self.inputs_dtype)
        segmentation = segmentation.to(self.outputs_dtype)

        return exam.unsqueeze(0), segmentation


class PadDepth(object):
    def __init__(self, where: str = "end", total_depth: int = 120, fill: int = 0) -> None:
        self.where = where
        self.total_depth = total_depth
        self.fill = fill

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        original_depth = image.shape[0]
        if original_depth > self.total_depth:
            raise ValueError(f"Image has depth {original_depth}; can't be padded down to {self.total_depth}")

        if self.where == "end":
            pad_dims = [0, 0, 0, 0, 0, self.total_depth - original_depth]
        elif self.where == "beginning":
            pad_dims = [0, 0, 0, 0, 0, self.total_depth - original_depth]
        else:
            assert self.where in ["beginning", "end"]
        return torch.nn.functional.pad(image, pad_dims, value=self.fill)


class Interpolate(object):
    def __init__(self, size: Tuple[int, int, int] = None, mode: str = "nearest") -> None:
        self.size = size
        self.mode = mode

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = image.unsqueeze(0).unsqueeze(0)
        image = torch.nn.functional.interpolate(image, size=self.size, mode=self.mode)
        image = image.squeeze(0).squeeze(0)

        return image


class SegmentationDataset2D(torch.utils.data.Dataset):
    def __init__(
        self,
        subject_folders: List[str],
        transforms_3d: Optional[List[Callable[[torch.tensor], torch.tensor]]] = None,
        transforms_2d: Optional[List[Callable[[torch.tensor], torch.tensor]]] = None,
        n_before: int = 3,
        n_after: int = 3,
    ) -> None:
        self.dataset_3d = SegmentationDataset3D(subject_folders, transforms_3d)

        self.transforms_2d = transforms_2d
        self.n_before = n_before
        self.n_after = n_after

        self.n_slices = self._get_n_slices()
        self.n_subslices = self.n_slices - self.n_before - self.n_after

        assert len(self) > 0, "n_before + n_after too large"

    def __len__(self) -> int:
        return self.n_subslices * len(self.dataset_3d)

    def _get_n_slices(self) -> int:
        return self.dataset_3d[0][0].shape[1]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        index_3d = index // self.n_subslices
        index_2d = index % self.n_subslices

        exam, segmentation = self.dataset_3d[index_3d]

        exam = exam[0, index_2d : index_2d + self.n_before + 1 + self.n_after, :, :]
        segmentation = segmentation[index_2d, :, :]

        if self.transforms_2d is not None:
            for transform in self.transforms_2d:
                exam = transform(exam)
                segmentation = transform(segmentation)

        return exam, segmentation


def get_random_split(
    dataset_list: List[str], train_proportion: float = 0.8, seed: int = 0
) -> Tuple[List[str], List[str]]:
    torch.manual_seed(seed)
    train_size = int(len(dataset_list) * train_proportion)
    val_size = len(dataset_list) - train_size
    train_subjects, val_subjects = torch.utils.data.random_split(dataset_list, (train_size, val_size))
    torch.manual_seed(torch.initial_seed())
    return train_subjects, val_subjects


def get_nrrd_scale(nrrd_header: OrderedDict) -> List[float]:
    """Return the scale for each direction coordinate in mm."""
    # space_directions = np.transpose(nrrd_header.direction) * nrrd_header.spacing
    space_directions = nrrd_header["space directions"]
    return [
        np.linalg.norm(space_directions[0]),
        np.linalg.norm(space_directions[1]),
        np.linalg.norm(space_directions[2]),
    ]


def get_nonzero_volume(nrrd_data: np.ndarray, space_directions: np.ndarray) -> float:
    """Return the volume (in ml) of the nonzero voxels in nrrd_data."""
    volume = np.count_nonzero(nrrd_data) * np.sqrt(
        (space_directions[0] @ space_directions[0])
        * (space_directions[1] @ space_directions[1])
        * (space_directions[2] @ space_directions[2])
    )
    volume /= 1000.0
    return volume


def get_mean_and_variance_of_segmented_color(exam_data: np.ndarray, seg_data: np.ndarray) -> Tuple[float, float]:
    """Return the mean and the variance of exam_data voxels which correspond to a nonzero voxel in seg_data.
    Assumes that the dimensions match.
    """
    seg_data = seg_data != 0
    mean = np.mean(exam_data, where=seg_data)
    variance = np.var(exam_data, where=seg_data)
    return mean, variance


def get_scaled_gradient(data: np.ndarray, scale: np.array, margin_size: int = 2) -> Tuple[np.ndarray]:
    """Given the the array of a segmentation or exam and the correct scale, return the corresponding gradient.
    Force voxels outside the margin to 0 and scale each direction according to the header data.
    """
    grad = np.gradient(data)
    mask = np.zeros(grad[0].shape)
    mask[
        margin_size : mask.shape[0] - margin_size,
        margin_size : mask.shape[1] - margin_size,
        1 : mask.shape[1] - margin_size,
    ] = 1

    grad[0] = np.multiply(grad[0], mask / scale[0])
    grad[1] = np.multiply(grad[1], mask / scale[1])
    grad[2] = np.multiply(grad[2], mask / scale[2])

    return grad


def _get_exam_gradient(exam_data: np.ndarray, exam_header: OrderedDict, margin_size: int = 2) -> Tuple[np.ndarray]:
    """Return the exam gradient.
    Force voxels outside the margin to 0 and scale each direction according to the header data.
    """
    grad = np.gradient(exam_data)
    mask = np.zeros(grad[0].shape)
    mask[
        margin_size : mask.shape[0] - margin_size,
        margin_size : mask.shape[1] - margin_size,
        1 : mask.shape[1] - margin_size,
    ] = 1
    scale = get_nrrd_scale(exam_header)

    grad[0] = np.multiply(grad[0], mask / scale[0])
    grad[1] = np.multiply(grad[1], mask / scale[1])
    grad[2] = np.multiply(grad[2], mask / scale[2])

    return grad


def get_exam_gradient(exam_data: np.ndarray, exam_header: OrderedDict, margin_size: int = 2) -> Tuple[np.ndarray]:
    """Return the exam gradient.
    Force voxels outside the margin to 0 and scale each direction according to the header data.
    """
    scale = get_nrrd_scale(exam_header)
    return get_scaled_gradient(exam_data, scale, margin_size=margin_size)


def get_1d_convex_hull(vec: np.ndarray) -> np.ndarray:
    """Naive implementation of convex hull.

    Example: get_1d_convex_hull(np.array([0, 1, 0, 0, 1, 0])) -> array([0, 1, 1, 1, 1, 0])
    """
    first = -1
    last = 0

    i = 0
    while first == -1 and i < len(vec):
        if vec[i] != 0:
            first = i
        i += 1

    last = first
    while i < len(vec):
        if vec[i] != 0:
            last = i
        i += 1

    out = np.zeros(vec.shape)
    out[first : last + 1] = 1
    return out


def get_3d_convex_hull(seg_data: np.ndarray) -> np.ndarray:
    """Return an approximation of a convex hull.
    Does the same as it's 1d counterpart, for all colluns parallel to each of the 3 axis.
    """

    out = seg_data
    for i, j in itertools.product(range(seg_data.shape[0]), range(seg_data.shape[1])):
        out[i, j, :] = get_1d_convex_hull(out[i, j, :])

    for i, j in itertools.product(range(seg_data.shape[1]), range(seg_data.shape[2])):
        out[:, i, j] = get_1d_convex_hull(out[:, i, j])

    for i, j in itertools.product(range(seg_data.shape[0]), range(seg_data.shape[2])):
        out[i, :, j] = get_1d_convex_hull(out[i, :, j])

    return out


def get_dataset_as_array(dataset: torch.utils.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    model_inputs, model_targets = next(
        iter(torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False))
    )

    return model_inputs.numpy(), model_targets.numpy()


def get_md5_from_filepath(path: str) -> str:
    """Return md5 of file contents."""
    f = open(path, "rb")
    md5_hash = hashlib.md5()
    md5_hash.update(f.read())
    return md5_hash.hexdigest()


def get_subject_id_from_md5s(md5_exam: str, md5_segmentation: str, enable_basename_checks: bool = True) -> str:
    """Return the subject id, given the md5s of the exam and of the segmentation

    Let abcde.. be the first half of the md5 of the exam, and
        12345 be the first half of the md5 of the segmenatation.

    Then the subject id would be
        a1b2c4d4e5...

    This format is useful, because we can find the exam and the segmentation
    pair that correspond to this subject without storing any additional data.

    Important note: according to this definition, a subject is a pair (exam,
    segmentation), and not the person itself.

    Here is an example of this distinction:
        (1) get_subject_id(exam.nrrd, segmentation.nrrd)
        (2) get_subject_id(segmentation.nrrd, exam.nrrd)
    (1) and (2) are ids for different subjects, (1) is the id of a valid
    subject and (2) is the id of an invalid subject.

    """
    if len(md5_exam) != 32 or len(md5_segmentation) != 32:
        raise ValueError("not a valid md5, lenght should be 32")
    return "".join("".join([x, y]) for x, y in zip(md5_exam, md5_segmentation))[:32]


class MoreThanOnePairFound(Exception):
    pass


def get_pool_paths_from_subject_id(subject_id: str) -> Tuple[str, str]:
    subject_id_chars = list(subject_id)
    exam_md5 = "".join(subject_id_chars[i] for i in range(len(subject_id_chars)) if i % 2 == 0)
    seg_md5 = "".join(subject_id_chars[i] for i in range(len(subject_id_chars)) if i % 2 == 1)
    exam_path = glob.glob(PATH_POOL_OF_NRRDS + "/" + exam_md5 + "*")
    seg_path = glob.glob(PATH_POOL_OF_NRRDS + "/" + seg_md5 + "*")

    if len(exam_path) > 1 or len(seg_path) > 1:
        raise MoreThanOnePairFound("More than one pair in the pool matching subject-id=" + subject_id)

    return exam_path[0], seg_path[0]


def get_subject_id_from_filepaths(path_exam: str, path_segmentation: str, enable_basename_checks: bool = True) -> str:
    """Return an unique identifier for the subject."""
    if enable_basename_checks:
        basename_exam = os.path.basename(path_exam)
        basename_segmentation = os.path.basename(path_segmentation)
        if basename_exam not in EXAM_BASENAMES:
            raise ValueError(
                basename_exam + " is not in our set EXAM_BASENAMES=" + str(EXAM_BASENAMES) + " of valid basenames."
            )
        if basename_segmentation not in SEGMENTATION_BASENAMES:
            raise ValueError(
                basename_segmentation
                + " is not in our set SEGMENTATION_BASENAMES="
                + str(SEGMENTATION_BASENAMES)
                + " of valid basenames."
            )

    md5_exam = get_md5_from_filepath(path_exam)
    md5_segmentation = get_md5_from_filepath(path_segmentation)
    return get_subject_id_from_md5s(md5_exam, md5_segmentation)


def get_random_error_filename() -> str:
    """Return something similar to 'error_f055ff64d484cbefcbc6b5eef79761c2ff57a634d3a1b105fabe944a38c52ebd'.
    Useful for safe file operations, since rewrite in POSIX is atomic.
    """
    randomness = token_hex(32)
    return "error_" + randomness


def safe_copy(orig: str, dest: str) -> None:
    """Perform a safe copy, that writes error_* in the dest folder in case of failure.

    For POSIX compliant systems, os.rename() is atomic, as long as the system
    doesn't crash, so we can copy to some file named error_<randomness> and
    rename when done.
    """
    dest_dir = os.path.dirname(dest)
    temp_dest = dest_dir + "/" + get_random_error_filename()
    shutil.copyfile(orig, temp_dest)
    os.rename(temp_dest, dest)


def horizontal_flip_3d(image: torch.Tensor) -> torch.Tensor:
    """Apply horizontal flip as an augmentation strategy to the 3d image (image)"""
    # convert to 4d tensor to add channels
    image = image.unsqueeze(0)
    # Creating transformation
    h_flip = tio.RandomFlip(axes=(0,), flip_probability=1)
    # Applying transformation and convert to 3d tensor
    t_image = h_flip(image)[0, :, :, :]
    return t_image


def random_bias_magnect_field_3d(image: torch.Tensor) -> torch.Tensor:
    """Add a random MRI bias field artifact to the 3d image (image)"""
    # convert to 4d tensor to add channels
    image = image.unsqueeze(0)
    # Creating transformation
    MRI_bias = tio.RandomBiasField(order=3)
    # Applying transformation and convert to 3d tensor
    t_image = MRI_bias(image)[0, :, :, :]
    return t_image


def random_ghosting_3d(image: torch.Tensor) -> torch.Tensor:
    """Add random MRI ghosting artifact to the 3d image (image)"""
    # convert to 4d tensor to add channels
    image = image.unsqueeze(0)
    # Creating transformation
    ghost = tio.RandomGhosting(num_ghosts=(1, 5), axes=(1, 2), intensity=(0, 1))
    # Applying transformation and convert to 3d tensor
    t_image = ghost(image)[0, :, :, :]
    return t_image


def seg_voxel_coords_to_exam_voxel_coords(
    exam_header: OrderedDict, seg_header: OrderedDict, voxel_coordinates_seg: np.ndarray
) -> np.ndarray:
    """Given segmentation coordinates in voxel space, return the corresponding coordinates in exam space."""

    matrix_exam = exam_header["space directions"].T
    exam_origin = np.array(exam_header["space origin"])
    inv_matrix_exam = np.linalg.inv(matrix_exam)
    matrix_seg = seg_header["space directions"].T
    seg_origin = np.array(seg_header["space origin"])
    real_coordinates_seg = (matrix_seg @ voxel_coordinates_seg) + seg_origin
    voxel_coordinates_exam = np.rint((inv_matrix_exam @ (real_coordinates_seg - exam_origin))).astype(int)

    return voxel_coordinates_exam


# filters for create-dataset_segmentation


def seg_to_exam_affine_transformatiion_is_identity(
    exam_data: np.ndarray, exam_header: OrderedDict, seg_data: np.ndarray, seg_header: OrderedDict
) -> bool:
    """Return true if affine transformation of the headers are compatible.

    Check if affine transfomations are the same. We are using the fact that
    voxel coordinates align to the grid to avoid any false negavites due to
    roundig errors.
    """
    e = np.array(exam_data.shape) - [1, 1, 1]

    return (
        np.array_equal(seg_voxel_coords_to_exam_voxel_coords(exam_header, seg_header, [0, 0, 0]), [0, 0, 0])
        and np.array_equal(seg_voxel_coords_to_exam_voxel_coords(exam_header, seg_header, [e[0], 0, 0]), [e[0], 0, 0])
        and np.array_equal(seg_voxel_coords_to_exam_voxel_coords(exam_header, seg_header, [0, e[1], 0]), [0, e[1], 0])
        and np.array_equal(seg_voxel_coords_to_exam_voxel_coords(exam_header, seg_header, [0, 0, e[2]]), [0, 0, e[2]])
    )


def voxel_coords_from_seg_to_exam_fall_inside_bounds(
    exam_data: np.ndarray, exam_header: OrderedDict, seg_data: np.ndarray, seg_header: OrderedDict
) -> bool:
    """Return true if affine transformations that take points from seg to exam are inside the bounds of the exam."""
    seg_max_coords = np.array(seg_data.shape) - [1, 1, 1]
    sm = seg_max_coords
    exam_max_coords = np.array(exam_data.shape) - [1, 1, 1]

    for x, y, z in itertools.product(range(2), range(2), range(2)):
        exam_point = seg_voxel_coords_to_exam_voxel_coords(exam_header, seg_header, [x * sm[0], y * sm[1], z * sm[2]])
        for i in range(3):
            if exam_point[i] < 0 or exam_point[i] > exam_max_coords[i]:
                return False
    return True


def get_dimensional_consistency(
    exam_data: np.ndarray,
    exam_header: OrderedDict,
    seg_data: np.ndarray,
    seg_header: OrderedDict,
) -> bool:
    return (
        np.array_equal(exam_data.shape, np.array(exam_header["sizes"]))
        and np.array_equal(seg_data.shape, np.array(seg_header["sizes"]))
        and np.array_equal(exam_data.shape, seg_data.shape)
    )


def linear_part_of_seg_to_exam_transformation_is_identity(
    exam_data: np.ndarray, exam_header: OrderedDict, seg_data: np.ndarray, seg_header: OrderedDict
) -> bool:
    """Return true if the linear part of the affine transfomration that take points from seg to exam is the identity matrix."""
    matrix_exam = exam_header["space directions"].T
    try:
        inv_matrix_exam = np.linalg.inv(matrix_exam)
    except np.linalg.LinAlgError:
        return False
    matrix_seg = seg_header["space directions"].T

    seg_max_coords = np.array(seg_data.shape) - [1, 1, 1]
    real_coordinates_seg = matrix_seg @ seg_max_coords
    voxel_coordinates_exam = np.rint(inv_matrix_exam @ real_coordinates_seg).astype(int)
    return np.array_equal(voxel_coordinates_exam, seg_max_coords)


# fixes for create-dataset_segmentation


def fix_segmentation_translation(
    exam_data: np.ndarray, exam_header: OrderedDict, seg_data: np.ndarray, seg_header: OrderedDict
) -> Tuple[np.ndarray, OrderedDict]:
    """Given that the headers are correct, return a new segment with correct dimensions and translation.
    This fix does not check for out-of-bounds errors.
    """

    matrix_exam = exam_header["space directions"].T
    exam_origin = exam_header["space origin"]
    inv_matrix_exam = np.linalg.inv(matrix_exam)
    seg_origin = seg_header["space origin"]

    # computes origin and endpoint in voxel exam space
    origin = np.rint((inv_matrix_exam @ (seg_origin - exam_origin))).astype(int)
    endpoint = origin + seg_data.shape

    new_seg_data = np.zeros(exam_data.shape)
    new_seg_data[origin[0] : endpoint[0], origin[1] : endpoint[1], origin[2] : endpoint[2]] = seg_data

    return new_seg_data, exam_header


def get_id_2_pool_paths(id_list: List[str]) -> Dict:
    """Given a subject id list, return a dictionary mapping ids to paths in the pool.

    The return format is the following: [path_exam, path_segmentation].
    """
    pool_paths = glob.glob(PATH_POOL_OF_NRRDS + "/*.nrrd")
    half_md5_2_pool_paths = {os.path.basename(p).split(".")[0][:16]: p for p in pool_paths}

    half_exam_md5 = [
        "".join(subject_id_chars[i] for i in range(len(subject_id_chars)) if i % 2 == 0) for subject_id_chars in id_list
    ]
    half_seg_md5 = [
        "".join(subject_id_chars[i] for i in range(len(subject_id_chars)) if i % 2 == 1) for subject_id_chars in id_list
    ]

    id_2_pool_paths = {
        id_list[i]: [half_md5_2_pool_paths[half_exam_md5[i]], half_md5_2_pool_paths[half_seg_md5[i]]]
        for i in range(len(id_list))
    }

    return id_2_pool_paths


def save_dataset_2d_hdf(raw_ids, path: str) -> None:
    with h5py.File(path, mode="w") as file:

        for subject_id in tqdm(raw_ids.keys(), desc="saving dataset"):

            path_to_exam = os.path.join(raw_ids[subject_id], "1.nrrd")
            path_to_segmentation = os.path.join(raw_ids[subject_id], "2.nrrd")
            exam_data, exam_header = nrrd.read(path_to_exam)
            segmentation_data, segmentation_header = nrrd.read(path_to_segmentation)
            segmentation_data = (segmentation_data != 0).astype(segmentation_data.dtype)

            subject_group = file.create_group(f"{subject_id}")
            subject_group.create_dataset("exam_transform", data=exam_header["space directions"], track_times=False)
            subject_group.create_dataset("exam_origin", data=exam_header["space origin"], track_times=False)
            subject_group.create_dataset(
                "segmentation_transform", data=segmentation_header["space directions"], track_times=False
            )
            subject_group.create_dataset(
                "segmentation_origin", data=segmentation_header["space origin"], track_times=False
            )

            slices_group = subject_group.create_group("slices")
            for i in range(exam_data.shape[2]):
                slice_group = slices_group.create_group(f"{i}")

                slice_group.create_dataset("exam", data=exam_data[..., i], **hdf5plugin.Blosc(), track_times=False)
                slice_group.create_dataset(
                    "segmentation", data=segmentation_data[..., i], **hdf5plugin.Blosc(), track_times=False
                )


def load_whole_subject(hdffile: h5py.File, subject: str) -> Tuple[np.ndarray, OrderedDict, np.ndarray, OrderedDict]:
    subject_group = hdffile[subject]
    slices_group = subject_group["slices"]
    slice_keys = sorted(list(slices_group.keys()), key=int)

    exam_data = np.stack([slices_group[i]["exam"] for i in slice_keys], axis=2)
    segmentation_data = np.stack([slices_group[i]["segmentation"] for i in slice_keys], axis=2)
    exam_header = OrderedDict()
    exam_header["space directions"] = subject_group["exam_transform"]
    exam_header["space origin"] = subject_group["exam_origin"]
    segmentation_header = OrderedDict()
    segmentation_header["space directions"] = subject_group["segmentation_transform"]
    segmentation_header["space origin"] = subject_group["segmentation_origin"]

    return exam_data, exam_header, segmentation_data, segmentation_header


def get_subject_id_from_raw_path(path_to_raw_subject: str) -> str:
    path_to_exam = os.path.join(path_to_raw_subject, "1.nrrd")
    path_to_segmentation = os.path.join(path_to_raw_subject, "2.nrrd")

    exam_md5 = get_md5_from_filepath(path_to_exam)
    segmentation_md5 = get_md5_from_filepath(path_to_segmentation)

    subject_id = get_subject_id_from_md5s(exam_md5, segmentation_md5)

    return subject_id


class IterIdExamTarget:
    def __init__(self, dataset_path: str, *, metadata: bool = False):
        self.file = h5py.File(dataset_path, "r")
        self.subjects = self.file.keys()
        self.metadata = metadata

    def __iter__(
        self,
    ) -> Iterator[
        Union[
            Tuple[str, np.ndarray, np.ndarray],
            Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ]
    ]:
        for subject in self.subjects:
            subject_group = self.file[subject]
            slices_group = subject_group["slices"]
            n = len(slices_group)

            out = (
                subject,
                np.array([np.asarray(slices_group[f"{slice}"]["exam"]) for slice in range(n)]).transpose(2, 0, 1),
                np.array([np.asarray(slices_group[f"{slice}"]["segmentation"]) for slice in range(n)]).transpose(
                    2, 0, 1
                ),
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
