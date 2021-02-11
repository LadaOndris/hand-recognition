from src.utils.paths import MSRAHANDGESTURE_DATASET_DIR
from src.utils.plots import plot_joints, plot_joints_only
import numpy as np
import glob
import struct


def read_image(file_path: str):
    with open(file_path, 'rb') as f:
        total_width, total_height = struct.unpack('ii', f.read(4 * 2))
        left, top, right, bottom = struct.unpack('i' * 4, f.read(4 * 4))
        width = right - left
        height = bottom - top

        image_data = f.read()
        values = len(image_data) // 4
        image = struct.unpack(F"{values}f", image_data)
        image = np.array(image)
        image = image.reshape([height, width])
        return image, np.array([left, top, right, bottom])


def read_images(path: str):
    image_file_paths = sorted(path.glob('*.bin'))
    images_and_boundaries = np.array([read_image(file_path) for file_path in image_file_paths])
    images = images_and_boundaries[..., 0]
    global_boundaries = np.stack(images_and_boundaries[..., 1])
    centered_joints, labels = load_joints(path.joinpath('joint.txt'), gesture=-1)
    local_joints = transform_joints_to_local(centered_joints, global_boundaries)
    return images, local_joints


def transform_joints_to_local(joints, global_boundaries):
    left, top, right, bottom = np.hsplit(global_boundaries, 4)
    image_center = (160, 120)
    # it looks like the joint coordinates are y,x,z instead of x,y,z

    # to global coordinates
    joints[..., 0] += image_center[0]
    joints[..., 1] = image_center[1] - joints[..., 1]
    # plot_joints_only(None, joints[0])

    # to local coordinates
    joints[..., 0] -= left
    joints[..., 1] -= top
    return joints


def load_joints(joints_file: str, gesture: str) -> np.ndarray:
    """
    Parameters
    ----------
    joints_file string : A path to the file containing joint annotations.
    gesture     string : Gesture label. It is not included inside the
        annotations file, so it needs to be passed as a parameter.
        It is used to create labels.

    Returns
    -------
    Joint annotations and corresponding labels.
    """
    joints = np.genfromtxt(joints_file, skip_header=1, dtype=np.float64)
    joints = np.reshape(joints, (-1, 21, 3))
    labels = np.full(shape=(joints.shape[0]), fill_value=gesture)
    return joints, labels


def load_dataset() -> np.ndarray:
    path = MSRAHANDGESTURE_DATASET_DIR.joinpath(F"P0/")
    files = path.iterdir()
    gesture_names = []
    joints, labels = None, None

    for file in files:
        gesture_name = file.stem
        j, l = load_joints(file.joinpath('joint.txt'), gesture_name)
        if joints is None:
            joints, labels = j, l
        else:
            joints = np.concatenate((joints, j))
            labels = np.concatenate((labels, l))
        gesture_names.append(gesture_name)
    return np.unique(gesture_names), joints, labels


if __name__ == '__main__':
    images, joints = read_images(MSRAHANDGESTURE_DATASET_DIR.joinpath('P0/5'))
    idx = 8
    plot_joints(images[idx], joints[idx])

