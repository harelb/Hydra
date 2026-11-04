import pathlib

import imageio.v3 as iio
import numpy as np

import hydra_image_store  # noqa: F401 - ADT4 shim: puts hydra_python on sys.path


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def make_T(R=None, t=(0, 0, 0)):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = t
    return T


def write_image(path, w=8, h=6, mode="RGB", value=128):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "I;16":
        arr = np.full((h, w), value, dtype=np.uint16)
    elif mode == "L":
        arr = np.full((h, w), value, dtype=np.uint8)
    else:
        arr = np.full((h, w, 3), value, dtype=np.uint8)
    iio.imwrite(path, arr)
    return path


BODY_T_SENSOR = make_T(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.0]]), (0.4, 0.0, 0.1))
