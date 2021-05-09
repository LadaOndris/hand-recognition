import glob
import os
import struct
from typing import Tuple

import numpy as np
import sklearn
import tensorflow as tf

from src.acceptance.base import fingers_length, hand_orientation, joint_relation_errors, transform_orientation_to_2d, \
    vectors_angle
from src.utils.camera import Camera
from src.utils.paths import DOCS_DIR, MSRAHANDGESTURE_DATASET_DIR
from src.utils.plots import plot_image_with_skeleton, plot_skeleton_with_jre


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
        image = image.reshape([height, width, 1])
        return image, np.array([left, top, right, bottom])


def read_images(path: str):
    image_file_paths = sorted(path.glob('*.bin'))
    images_and_bboxes = np.array([read_image(file_path) for file_path in image_file_paths], dtype=np.object)
    images = images_and_bboxes[..., 0]
    bbox_coords = np.stack(images_and_bboxes[..., 1])
    centered_joints = load_joints(path.joinpath('joint.txt'))
    return images, bbox_coords, centered_joints


def load_joints(joints_file: str) -> np.ndarray:
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
    joints = np.genfromtxt(joints_file, skip_header=1, dtype=np.float32)
    joints = np.reshape(joints, (-1, 21, 3))

    joints[..., 2] *= -1  # Make Z axis positive
    joints[..., 1] *= -1  # Reverse pinhole's camera inverted image (invert around center)
    return joints


def get_subject_dirs():
    all_files = MSRAHANDGESTURE_DATASET_DIR.iterdir()
    return [file for file in all_files if file.is_dir()]


def get_gesture_files(subject_dirs):
    files = []
    for path in subject_dirs:
        files += path.iterdir()
    return files


def load_dataset(shuffle=False) -> np.ndarray:
    subject_dirs = get_subject_dirs()
    gesture_files = get_gesture_files(subject_dirs)
    gesture_names = []
    joints, labels = None, None

    for file in gesture_files:
        gesture_name = file.stem
        j = load_joints(file.joinpath('joint.txt'))
        l = np.full(shape=(j.shape[0]), fill_value=gesture_name, dtype='<U3')
        if joints is None:
            joints, labels = j, l
        else:
            joints = np.concatenate((joints, j))
            labels = np.concatenate((labels, l))
        gesture_names.append(gesture_name)

    if shuffle:
        joints, labels = sklearn.utils.shuffle(joints, labels)
    return np.unique(gesture_names), joints, labels


class MSRADataset:

    def __init__(self, dataset_path, batch_size, shuffle=True):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.shuffle = shuffle

        subject_folders = self._get_subject_folders()
        self.train_subjects = subject_folders[:-1]
        self.test_subjects = subject_folders[-1:]

        train_images, train_joints = self.get_records(self.train_subjects)
        test_images, test_joints = self.get_records(self.test_subjects)
        self.train_size = len(train_joints)
        self.test_size = len(test_joints)

        self.num_train_batches = int(self.train_size // self.batch_size)
        self.num_test_batches = int(self.test_size // self.batch_size)

        self.train_dataset = self._build_dataset(train_images, train_joints)
        self.test_dataset = self._build_dataset(test_images, test_joints)
        self.records = None

    def _get_subject_folders(self):
        all_files = self.dataset_path.iterdir()
        return [file for file in all_files if file.is_dir()]

    def _build_dataset(self, images, joints):
        ds = tf.data.Dataset.from_tensor_slices((images, joints))
        if self.shuffle:
            ds = ds.shuffle(buffer_size=len(joints), reshuffle_each_iteration=True)
        ds = ds.repeat()
        ds = ds.map(self._read_image)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=1)
        return ds

    def get_records(self, subject_folders):
        gesture_folders = []
        for f in subject_folders:
            folders = f.iterdir()
            gesture_folders += folders
        images = []  # of shape [n_subjects * n_gestures * 500, 2]
        joints = None
        for folder in gesture_folders:
            folder_images, folder_joints = self._records_from_gesture_folder(folder)
            images += folder_images
            if joints is None:
                joints = folder_joints
            else:
                joints = np.concatenate([joints, folder_joints])
        return images, joints

    def _records_from_gesture_folder(self, path):
        image_file_paths = sorted(glob.glob(os.path.join(str(path), '*.bin')))
        joints = load_joints(path.joinpath('joint.txt'))
        return image_file_paths, joints

    def _read_image(self, image_file, joints):
        image, bbox = tf.numpy_function(read_image, [image_file], Tout=(tf.float64, tf.int64))
        left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]  # tf.unstack(bbox[0], axis=-1)
        image = tf.pad(image, [[top, 240 - bottom], [left, 320 - right], [0, 0]])
        image = tf.cast(image, dtype=tf.float32)
        # image = tf.expand_dims(image, 0)
        # image = tf.RaggedTensor.from_tensor(image, ragged_rank=2)
        return image, joints

    def _squeeze_image_dimension(self, images, bboxes, joints):
        return tf.squeeze(images, axis=1), bboxes, joints


def plot_hands():
    images, bbox_coords, joints = read_images(MSRAHANDGESTURE_DATASET_DIR.joinpath('P0/5'))
    images2, bbox_coords2, joints2 = read_images(MSRAHANDGESTURE_DATASET_DIR.joinpath('P0/2'))

    cam = Camera('msra')
    idx = 8
    idx2 = idx + 301
    joints1_2d = cam.world_to_pixel(joints[idx])
    plot_image_with_skeleton(images[idx], joints1_2d[..., :2] - bbox_coords[idx, ..., :2], show_fig=False)
    joints2_2d = cam.world_to_pixel(joints2[idx])
    plot_image_with_skeleton(images2[idx], joints2_2d[..., :2] - bbox_coords2[idx, ..., :2], show_fig=False)

    hand1 = (images[idx], bbox_coords[idx], joints[idx])
    hand2 = (images2[idx2], bbox_coords2[idx2], joints2[idx2])

    fig1 = DOCS_DIR.joinpath('figures/msra_jre_1.png')
    fig2 = DOCS_DIR.joinpath('figures/msra_jre_2.png')
    plot_two_hands_diff(hand1, hand2, cam, show_norm=True, show_joint_errors=True, fig_location=None)
    plot_two_hands_diff(hand2, hand1, cam, show_norm=True, show_joint_errors=True, fig_location=None)

    hand1_orientation, _ = hand_orientation(joints[idx])
    hand2_orientation, _ = hand_orientation(joints2[idx2])
    orientation_diff_rad = vectors_angle(hand1_orientation, hand2_orientation)
    orientation_diff_deg = np.rad2deg(orientation_diff_rad)
    # print(F"Hand1 orientation: {hand1_orientation:.2f}")
    # print(F"Hand2 orientation: {hand2_orientation:.2f}")
    angle = "{:0.2f}".format(orientation_diff_deg)
    print(F"\nOrientation difference as an angle: {angle} degrees\n")
    # plot_norm(joints[idx])


def try_fingers_lengths():
    """
    Calls fingers_length function on a few
    images from the MSRA dataset.
    (For experimental purposes)
    """
    images, bbox_coords, joints = read_images(MSRAHANDGESTURE_DATASET_DIR.joinpath('P0/2'))
    lengths = fingers_length(joints[0:5])
    return lengths


def mean_finger_length():
    """
    Computes the average length of a finger
    in the MSRA dataset.
    Result: 65 mm

    Returns
    -------
    A scalar value in milimeters representing the
    mean finger length.
    """
    msra = MSRADataset(MSRAHANDGESTURE_DATASET_DIR, batch_size=64)
    it = iter(msra.train_dataset)
    lengths_train = finger_lengths_for_iter(it, msra.num_train_batches)
    it = iter(msra.test_dataset)
    lengths_test = finger_lengths_for_iter(it, msra.num_test_batches)
    lengths = np.concatenate([lengths_train, lengths_test])
    mean_length = np.mean(lengths)
    return mean_length


def finger_lengths_for_iter(batch_iterator, batches):
    lengths_arrays = []
    for i in range(batches):
        images, joints = next(batch_iterator)
        lengths = fingers_length(joints)
        lengths_arrays.append(lengths)
    lengths = np.concatenate(lengths_arrays)
    return lengths


def plot_two_hands_diff(hand1: Tuple[np.ndarray, np.ndarray, np.ndarray],
                        hand2: Tuple[np.ndarray, np.ndarray, np.ndarray],
                        cam: Camera,
                        show_norm=False, show_joint_errors=False,
                        fig_location=None, show_fig=True):
    """
    Plots an image with a skeleton of the first hand and
    shows major differences from the second hand.

    Parameters
    ----------
    hand1 A tuple of three ndarrays: image, bbox, joints
    hand2 A tuple of three ndarrays: image, bbox, joints
    show_norm   bool True if vectors of the hand orientations should be displayed.
    show_diffs  bool True if differences between the hand poses should be displayed.
    """
    # plot the first skeleton
    # plot the second skeleton
    # compute the error between these skeletons
    # aggregate the errors for each joint
    # display the error for each joint
    image1, bbox1, joints1 = hand1
    image2, bbox2, joints2 = hand2

    if show_joint_errors:
        errors = joint_relation_errors(np.expand_dims(joints1, axis=0), np.expand_dims(joints2, axis=0))[0, 0, :]
    else:
        errors = None
    joints1_2d = cam.world_to_pixel(joints1)
    # The image is cropped. Amend the joint coordinates to match the depth image
    joints1_2d = joints1_2d[..., :2] - bbox1[..., :2]

    if show_norm:
        mean, norm = _compute_orientation(joints1, bbox1, cam)
    else:
        mean = None
        norm = None
    plot_skeleton_with_jre(image1, joints1_2d, errors, mean_vec=mean, norm_vec=norm,
                           fig_location=fig_location, show_fig=show_fig)


def _compute_orientation(joints, bbox, cam: Camera):
    norm, mean = hand_orientation(joints)
    norm2d, mean2d = transform_orientation_to_2d(norm, mean, bbox, cam)
    return mean2d, norm2d


if __name__ == '__main__':
    # mean_finger_length()
    np.set_printoptions(precision=2)
    plot_hands()
    # msra = MSRADataset(MSRAHANDGESTURE_DATASET_DIR, batch_size=4)
    # it = iter(msra.train_dataset)
    # for images, bboxes, joints in it:
    #     print(images.shape, joints.shape)
    pass
