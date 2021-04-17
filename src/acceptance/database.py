"""
This file contains functions to scan user-defined gestures,
extract poses, and save them into a directory.
"""
from src.utils.logs import make_timestamped_dir
from src.utils.paths import USECASE_DATASET_DIR, USECASE_DATASET_JOINTS_PATH
from src.utils.live import generate_live_images
import numpy as np


class UsecaseDatabase:

    def __init__(self, subdir):
        self.subdir = subdir
        self.subdir_path = USECASE_DATASET_DIR.joinpath(subdir)
        self.hand_poses = None
        self.labels = None

    def scan_into_subdir(self):
        pass

    def load_from_subdir(self):
        self.hand_poses = None
        self.labels = None

    def get_label_by_index(self, hand_pose_index):
        if self.labels is None:
            raise Exception("No hand poses are loaded")
        if hand_pose_index < 0 or hand_pose_index >= np.shape(self.labels)[0]:
            raise Exception("Index out of bounds")
        return self.labels[hand_pose_index]


def scan_raw_images(scan_period: float):
    """
    Scans images in intervals specified by 'scan_period'.
    Also creates a new directory for the scanned images.
    """

    # Scan images
    live_image_generator = generate_live_images()

    # Save them to folder
    scan_dir = make_timestamped_dir(USECASE_DATASET_DIR)

    pass


def produce_hand_keypoints():
    # 1. load detection and pose estimation models
    # 2. read depth image
    # 3. pass the image to detection model
    # 4. pass the subimage to pose estimation model
    # profit
    pass


def load_gestures():
    annotations = np.genfromtxt(USECASE_DATASET_JOINTS_PATH, skip_header=0, dtype=np.str)
    filenames = annotations[:, 0]
    joints = np.reshape(annotations[:, 1:], (-1, 21, 3))
    return filenames, joints


if __name__ == '__main__':
    scan_raw_images(scan_period=5)
    # produce_hand_keypoints()
    # files, joints = load_gestures()
    pass
