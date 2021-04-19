from src.system.hand_position_estimator import HandPositionEstimator
from src.utils.camera import Camera
from src.utils.paths import USECASE_DATASET_DIR
from src.utils.live import generate_live_images
import numpy as np
from src.utils.logs import make_dir, get_current_timestamp


class UsecaseDatabaseScanner:

    def __init__(self, subdir, camera=Camera('SR305'), plot_estimation=True):
        self.subdir = subdir
        self.camera = camera
        self.subdir_path = USECASE_DATASET_DIR.joinpath(subdir)
        self.estimator = HandPositionEstimator(self.camera, 230, plot_estimation=plot_estimation)

    def scan_into_subdir(self, gesture_label, scan_period=1):
        """
        Scans images in intervals specified by 'scan_period',
        and saves estimated joints into a new file with current timestamp
        in a directory specified by subdir.
        """
        file_path = self._prepare_file(gesture_label)
        generator = generate_live_images()
        with open(file_path, 'a+') as file:
            self._scan_from_generator(generator, file)

    def _scan_from_generator(self, generator, file):
        for image in generator:
            joints_uvz = self.estimator.inference_from_image(image)
            if joints_uvz is None:
                continue
            joints_xyz = self.camera.pixel_to_world(joints_uvz)
            self._save_joints_to_file(file, joints_xyz)

    def _prepare_file(self, gesture_label):
        make_dir(self.subdir_path)
        timestamp = get_current_timestamp()
        timestamped_file = self.subdir_path.joinpath(F"{gesture_label}_{timestamp}.txt")
        return timestamped_file

    def _save_joints_to_file(self, file, joints):
        formatted_joints = self._format_joints(joints)
        file.write(F"{formatted_joints}\n")

    def _format_joints(self, joints):
        flattened_joints = np.flatten(joints)
        formatted_joints = ' '.join(flattened_joints)
        return formatted_joints
