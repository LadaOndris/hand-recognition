from src.utils.paths import MSRAHANDGESTURE_DATASET_DIR, DOCS_DIR
from src.utils.plots import plot_joints, plot_joints_only, plot_two_hands_diff, plot_joints_2d, plot_norm
from src.acceptance.base import rds_errors, hand_rotation, vectors_angle
from src.utils.camera import Camera
import sklearn
import numpy as np
import glob
import struct
import os
import tensorflow as tf


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
        # ds = ds.map(self._squeeze_image_dimension)
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
        return image, bbox, joints

    def _squeeze_image_dimension(self, images, bboxes, joints):
        return tf.squeeze(images, axis=1), bboxes, joints


def plot_hands():
    images, bbox_coords, joints = read_images(MSRAHANDGESTURE_DATASET_DIR.joinpath('P0/5'))
    images2, bbox_coords2, joints2 = read_images(MSRAHANDGESTURE_DATASET_DIR.joinpath('P0/2'))

    cam = Camera('msra')
    idx = 8
    idx2 = idx + 301
    joints1_2d = cam.world_to_pixel(joints[idx])
    plot_joints_2d(images[idx], joints1_2d[..., :2] - bbox_coords[idx, ..., :2], show_fig=False)
    joints2_2d = cam.world_to_pixel(joints2[idx])
    plot_joints_2d(images2[idx], joints2_2d[..., :2] - bbox_coords2[idx, ..., :2], show_fig=False)

    hand1 = (images[idx], bbox_coords[idx], joints[idx])
    hand2 = (images2[idx2], bbox_coords2[idx2], joints2[idx2])

    fig1 = DOCS_DIR.joinpath('figures/msra_jre_1.png')
    fig2 = DOCS_DIR.joinpath('figures/msra_jre_2.png')
    plot_two_hands_diff(hand1, hand2, cam, show_norm=True, show_joint_errors=True, fig_location=fig1)
    plot_two_hands_diff(hand2, hand1, cam, show_norm=True, show_joint_errors=True, fig_location=fig2)

    hand1_orientation, _ = hand_rotation(joints[idx])
    hand2_orientation, _ = hand_rotation(joints2[idx2])
    orientation_diff_rad = vectors_angle(hand1_orientation, hand2_orientation)
    orientation_diff_deg = np.rad2deg(orientation_diff_rad)
    # print(F"Hand1 orientation: {hand1_orientation:.2f}")
    # print(F"Hand2 orientation: {hand2_orientation:.2f}")
    angle = "{:0.2f}".format(orientation_diff_deg)
    print(F"\nOrientation difference as an angle: {angle} degrees\n")
    # plot_norm(joints[idx])


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    plot_hands()
    # msra = MSRADataset(MSRAHANDGESTURE_DATASET_DIR, batch_size=4)
    # it = iter(msra.train_dataset)
    # for images, bboxes, joints in it:
    #     print(images.shape, joints.shape)
    pass
