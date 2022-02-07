"""
Visualize a prediction as a 3D mesh.

We transform the prediction into a 3D mesh using marching cubes (namely, Lewiner's marching cubes).
"""

from argparse import ArgumentParser
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes

from src.utils.colors import PREDICTION

parser = ArgumentParser()
parser.add_argument("-p", "--prediction", type=Path, required=True, help="Path to prediction HDF")
parser.add_argument("-s", "--subject_id", required=True, help="Which subject to visualize")
parser.add_argument("-o", "--output", type=str, required=True, help="Where to write the output image to")
parser.add_argument("--edt", action="store_true", help="Use EDT transform on prediction")
args = parser.parse_args()

with h5py.File(args.prediction) as file:
    if args.subject_id not in file:
        print("available subjects:")
        for subject_id in file.keys():
            print(f" - {subject_id}")
    subject_group = file[args.subject_id]

    exam = subject_group["exam"][:]
    assert isinstance(exam, np.ndarray)
    prediction = subject_group["prediction"][:] >= 0.5
    assert isinstance(prediction, np.ndarray)

    if args.edt:
        prediction = distance_transform_edt(np.where(prediction >= 0.5, 0, 1))

    verts, faces, normals, values = marching_cubes(prediction, 0.5)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    mesh = Poly3DCollection(verts[faces], alpha=1.0, linewidths=0)

    ls = LightSource(azdeg=0, altdeg=65)
    normalsarray = np.array(
        [
            np.array(
                (
                    np.sum(normals[face[:], 0] / 3),
                    np.sum(normals[face[:], 1] / 3),
                    np.sum(normals[face[:], 2] / 3),
                )
                / np.sqrt(
                    np.sum(normals[face[:], 0] / 3) ** 2
                    + np.sum(normals[face[:], 1] / 3) ** 2
                    + np.sum(normals[face[:], 2] / 3) ** 2
                )
            )
            for face in faces
        ]
    )

    _min = np.min(ls.shade_normals(normalsarray, fraction=1.0))  # min shade value
    _max = np.max(ls.shade_normals(normalsarray, fraction=1.0))  # max shade value
    _diff = _max - _min
    _newMin = 0.3
    _newMax = 0.95
    _newdiff = _newMax - _newMin

    colourRGB = np.array(PREDICTION)
    rgbNew = np.array(
        [
            colourRGB * (_newMin + _newdiff * ((shade - _min) / _diff))
            for shade in ls.shade_normals(normalsarray, fraction=1.0)
        ]
    )
    mesh.set_facecolor(rgbNew)

    ax.add_collection3d(mesh)

    ax.set_xlim(0, prediction.shape[0])
    ax.set_ylim(0, prediction.shape[1])
    ax.set_zlim(0, prediction.shape[2])

    ax.set_axis_off()
    plt.tight_layout()

    plt.savefig(args.output, dpi=300)
