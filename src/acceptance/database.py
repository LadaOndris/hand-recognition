"""
This file contains functions to scan user-defined gestures,
extract poses, and save them into a directory.
"""
from src.position_estimation import HandPositionEstimator
from src.utils.camera import Camera
from src.utils.paths import USECASE_DATASET_DIR, USECASE_DATASET_JOINTS_PATH
from src.utils.live import generate_live_images
import numpy as np
from src.utils.logs import make_dir, get_current_timestamp


class UsecaseDatabase:

    def __init__(self, subdir):
        self.subdir = subdir
        self.subdir_path = USECASE_DATASET_DIR.joinpath(subdir)
        self.hand_poses = None
        self.labels = None

    def scan_into_subdir(self, scan_period=1):
        """
        Scans images in intervals specified by 'scan_period'.
        Also creates a new directory for the scanned images.
        """
        file_path = self.prepare_file()
        camera = Camera('SR305')
        estimator = HandPositionEstimator(camera, 230, plot_estimation=True)
        generator = generate_live_images()
        with open(file_path, 'a+') as file:
            for image in generator:
                joints_uvz = estimator.inference_from_image(image)
                joints_xyz = camera.pixel_to_world(joints_uvz)
                self.save_joints_to_file(file, joints_xyz)

    def prepare_file(self):
        make_dir(self.subdir_path)
        timestamp = get_current_timestamp()
        timestamped_file = self.subdir_path.joinpath(F"{timestamp}.txt")
        return timestamped_file

    def save_joints_to_file(self, file, joints):
        formatted_joints = self.format_joints(joints)
        file.write(F"{formatted_joints}\n")

    def format_joints(self, joints):
        flattened_joints = np.flatten(joints)
        formatted_joints = ' '.join(flattened_joints)
        return formatted_joints

    def load_from_subdir(self):
        self.hand_poses = None
        self.labels = None

    def get_label_by_index(self, hand_pose_index):
        if self.labels is None:
            raise Exception("No hand poses are loaded")
        if hand_pose_index < 0 or hand_pose_index >= np.shape(self.labels)[0]:
            raise Exception("Index out of bounds")
        return self.labels[hand_pose_index]


def load_gestures():
    annotations = np.genfromtxt(USECASE_DATASET_JOINTS_PATH, skip_header=0, dtype=np.str)
    filenames = annotations[:, 0]
    joints = np.reshape(annotations[:, 1:], (-1, 21, 3))
    return filenames, joints


# if __name__ == '__main__':
#     #scan_raw_images(scan_period=5)
#     # produce_hand_keypoints()
#     # files, joints = load_gestures()
#     pass
