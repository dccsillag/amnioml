"""
This script loads a trained neural network model from a given checkpoint, gets
its predictions over the validation or test set, and saves them in a HDF file.

Example usage:

python src/models/predict.py --use_gpu \
    -cp personal/models/small_unet/congo_small_unet_2021-11-18T08:28:08.179-03:00/checkpoints/last.ckpt
"""

import argparse
import importlib
import os
from typing import Dict

import h5py
import hdf5plugin
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from src.utils.general_utils import get_eval_folder_path, outpath
from src.utils.model_utils import HdfDataloader2d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-bf",
        "--output_to_base_folder",
        action="store_true",
        help="save output to base folder instead of personal folder",
    )
    parser.add_argument(
        "-cp",
        "--checkpoint_path",
        help="checkpoint path",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="use gpu to make predictions",
    )

    parser.add_argument(
        "-d",
        "--dataset_path_overwrite",
        default="",
        help="path to another hdf5 file",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="use test database instead of val",
    )
    args = parser.parse_args()

    run_name = os.path.basename(os.path.dirname(os.path.dirname(args.checkpoint_path)))
    try:
        model_name = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))["hyper_parameters"]["args"][
            "model"
        ]  # to read model_name without relying on folders
    except KeyError:
        print(
            "warning: the name of the model wasn't found on hyper_paramaters, "
            "trying to read it from the folder structure - this will likely go wrong..."
        )
        model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.checkpoint_path))))

    if model_name.startswith("smp"):
        model = (
            importlib.import_module("src.models.smp")
            .__getattribute__(model_name.split("_")[1])
            .load_from_checkpoint(args.checkpoint_path)
        )
    else:
        model_module = importlib.import_module(f"src.models.{model_name}.model")
        model = model_module.Model.load_from_checkpoint(args.checkpoint_path)  # type: ignore

    if args.use_gpu:
        model.to("cuda")
    else:
        model.to("cpu")
    model.freeze()
    model.eval()

    stride = model.args.get("channel_stride", 1)
    normalize_using_center = model.args.get("normalize_using_center", False)

    dataset_path = f"data/processed/{model.args['dataset']}"
    if args.test:
        dataset_path = f"{dataset_path}/test.hdf"
    else:
        dataset_path = f"{dataset_path}/val.hdf"

    if len(args.dataset_path_overwrite) >= 1:
        dataset_path = args.dataset_path_overwrite

    dataloader = HdfDataloader2d(
        dataset="",
        image_size=model.args["image_size"],
        num_workers=1,
        batch_size=1,
        n_slices=model.args["num_channels"],
        stride=stride,
        normalize_using_center=normalize_using_center,
    )
    torch_dataloader, dataset_id_with_slices = dataloader.get_dataloader_and_dataset_ids(
        dataset_path, batch_size=1, shuffle=False
    )

    subject_id_2_slices: Dict[str, Dict] = {}
    for i in range(len(dataset_id_with_slices)):
        subject_id, slice_i = dataset_id_with_slices[i]
        subject_id_2_slices.setdefault(subject_id, {})
        subject_id_2_slices[subject_id][int(slice_i)] = i

    # pred = ModelPrediction2d(model, dataloader, dataset_path, original_resolution=True)

    output_folder = get_eval_folder_path(model_name, run_name, args.output_to_base_folder, test_database=args.test)
    if len(args.dataset_path_overwrite) >= 1:
        output_folder += (
            f"/{os.path.basename(os.path.dirname(args.dataset_path_overwrite))}"
            f"-{os.path.splitext(os.path.basename(args.dataset_path_overwrite))[0]}"
        )

    print("output_folder=" + output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    out_path = os.path.join(
        output_folder, os.path.basename(os.path.splitext(args.checkpoint_path)[0]) + "_original_res.hdf5"
    )

    if os.path.exists(out_path):
        raise FileExistsError(f"{out_path} already exists")
    print(f"out_path={out_path}")

    with h5py.File(outpath(out_path), "w") as file:
        with h5py.File(dataset_path) as dataset_file:
            subject_ids = list(dataset_file.keys())

            for subject_id in tqdm(subject_ids, desc="saving"):
                subject_group = file.create_group(subject_id)
                subject_group.create_dataset(
                    "exam_transform", data=dataset_file[subject_id]["exam_transform"], track_times=False
                )
                subject_group.create_dataset(
                    "exam_origin", data=dataset_file[subject_id]["exam_origin"], track_times=False
                )
                subject_group.create_dataset(
                    "segmentation_transform", data=dataset_file[subject_id]["segmentation_transform"], track_times=False
                )
                subject_group.create_dataset(
                    "segmentation_origin", data=dataset_file[subject_id]["segmentation_origin"], track_times=False
                )

                exam = np.stack(
                    [
                        dataset_file[subject_id]["slices"][i]["exam"]
                        for i in sorted(list(dataset_file[subject_id]["slices"].keys()), key=int)
                    ],
                    axis=2,
                )
                segmentation = np.stack(
                    [
                        dataset_file[subject_id]["slices"][i]["segmentation"]
                        for i in sorted(list(dataset_file[subject_id]["slices"].keys()), key=int)
                    ],
                    axis=2,
                )
                subject_group.create_dataset("exam", data=exam, **hdf5plugin.Blosc(), track_times=False)
                subject_group.create_dataset("segmentation", data=segmentation, **hdf5plugin.Blosc(), track_times=False)
                assert exam.shape == segmentation.shape

                # make prediction
                slice_2_index = subject_id_2_slices[subject_id]
                pred = np.zeros(segmentation.shape, dtype=np.float32)
                iter_slices = list(slice_2_index.keys())
                for j in range(len(iter_slices)):
                    i = iter_slices[j]
                    with torch.no_grad():
                        s = model(torch_dataloader.dataset[slice_2_index[i]][0].unsqueeze(0).to(model.device.type)).to(
                            "cpu"
                        )
                        s = torch.sigmoid(s)
                        if (
                            segmentation.shape[0] > 0
                            and segmentation.shape[1] > 0
                            and (segmentation.shape[0] != s.shape[2] or segmentation.shape[1] != s.shape[3])
                        ):
                            s = torchvision.transforms.Resize(
                                (segmentation.shape[0], segmentation.shape[1]),
                                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                            )(s)
                        pred[:, :, i] = s.squeeze().squeeze().detach().numpy()

                subject_group.create_dataset("prediction", data=pred, **hdf5plugin.Blosc(), track_times=False)
