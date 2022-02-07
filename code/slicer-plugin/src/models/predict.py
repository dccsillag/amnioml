import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-cp",
    "--checkpoint_path",
    type=str,
    help="path to the model's checkpoint",
)

parser.add_argument(
    "-ep",
    "--exam_path",
    type=str,
    help="path to the exam (NRRD format)",
)

parser.add_argument(
    "-o",
    "--output_path",
    type=str,
    help="where to save the prediction (NRRD format)",
)

parser.add_argument(
    "--use_gpu",
    action="store_true",
    help="use the GPU to make the prediction, instead of the CPU",
)

parser.add_argument(
    "--cuda_is_available",
    action="store_true",
    help="echo True if cuda is avaiable, then exit",
)

args = parser.parse_args()

import torch
import sys
if args.cuda_is_available:
    print(torch.cuda.is_available())
    sys.exit(0)


import os
import torchvision
from torchvision.transforms import InterpolationMode
import numpy as np

# we need static import, so that pyinstaller can read the exe
import src.models.bio_unet.model as model_module

from tqdm.tk import trange

# from medpy.io import load, save

# Medpy's save may loose some meta-information (see
# https://loli.github.io/medpy/generated/medpy.io.save.save.html),
# so nrrd was chosen instead.
import nrrd

# Adapted from HdfDataset2d (in the main branch)
def get_slice(raw_exam:np.array, exam_transforms, slice_i: int, normalize_using_center:bool=False, n_slices:int=1, stride:int=1) -> torch.Tensor:

    half_n_slices = int((n_slices - 1) / 2)

    inputs_dtype = torch.float32

    exam = np.array(
        [
            raw_exam[:,:,slice_i + stride * (j - half_n_slices)]
            for j in range(2 * half_n_slices + 1)
        ],
        dtype=np.float32,
    )

    # Normalize to 0-1
    if not normalize_using_center:
        exam = exam / max(exam.max(), 1.0)

    # Convert to torch tensor
    exam = torch.from_numpy(np.array(exam))

    # Apply exam transforms
    if exam_transforms is not None:
        for transform in exam_transforms:
            exam = transform(exam)

    # Typecasting
    exam = exam.to(inputs_dtype)

    return exam



def get_prediction(
    exam:np.array,
    checkpoint_path: str,
    use_gpu: bool = False,
) -> np.array:

    # load the model
    model = model_module.Model.load_from_checkpoint((checkpoint_path))  # type: ignore
    if use_gpu:
        model.to("cuda")
    else:
        model.to("cpu")
    model.freeze()
    model.eval()


    # read the arguments
    stride = 1
    if "channel_stride" in model.args:
        stride = model.args["channel_stride"]

    normalize_using_center = False
    if "normalize_using_center" in model.args:
        normalize_using_center = model.args["normalize_using_center"]

    n_slices = 1
    if "num_channels" in model.args:
        n_slices = model.args["num_channels"]

    half_n_slices = int((n_slices - 1) / 2)

    image_size = model.args["image_size"]

    exam_transforms = [
        torchvision.transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)
    ]


    total_number_of_slices = exam.shape[2]

    prediction = np.zeros(exam.shape)

    for slice_i in trange(exam.shape[2]):

        # Only include slices that fall inside bounds
        if (
            0 <= int(slice_i) - half_n_slices * stride
            and int(slice_i) + half_n_slices * stride < total_number_of_slices
        ):
            slice_tensor = get_slice(exam, exam_transforms, slice_i, normalize_using_center, n_slices, stride)
            s = None
            with torch.no_grad():
                s = model(slice_tensor.unsqueeze(0).to(model.device.type)).to("cpu")
                s = torch.sigmoid(s)

            if exam.shape[0] > 0 and exam.shape[1] > 0 and (exam.shape[0] != s.shape[2] or exam.shape[1] != s.shape[3]):
                s = torchvision.transforms.Resize((exam.shape[0], exam.shape[1]), interpolation=InterpolationMode.BILINEAR)(s)

            prediction[:, :, slice_i] = s

    return prediction




if args.output_path == None or args.output_path == "":
    raise ValueError(
        "please specify an OUTPUT_PATH."
    )

if os.path.exists(args.output_path):
    raise FileExistsError(
        f"OUTPUT_PATH={args.output_path} already exists."
    )

# model_name is used to load the correct model
try:
    model_name = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))["hyper_parameters"]["args"][
        "model"
    ]  # to read model_name without relying on folders
except KeyError:
    print(
        "warning: the name of the model wasn't found on hyper_paramaters, trying to read it from the folder structure - this will likely go wrong..."
    )
    model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.checkpoint_path))))

exam, header = nrrd.read(args.exam_path)

# compute the prediction
prediction = get_prediction(
    exam,
    args.checkpoint_path,
    args.use_gpu,
)

# compression_level=9 runs several times slower, only saving about 10 to
# 15% of space.
nrrd.write(args.output_path, data=prediction, header=header,
        compression_level=1)
