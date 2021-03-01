from src.utils.logs import make_timestamped_dir
from src.utils.paths import USECASE_DATASET_DIR, USECASE_DATASET_JOINTS_PATH
from src.utils.live import generate_live_images
import numpy as np


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
