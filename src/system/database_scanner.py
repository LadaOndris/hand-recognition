import argparse

from src.system.hand_position_estimator import HandPositionEstimator
from src.utils.camera import Camera
from src.utils.paths import USECASE_DATASET_DIR
from src.utils.live import generate_live_images
import numpy as np
from src.utils.logs import make_dir, get_current_timestamp
import time


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
            self._scan_from_generator(generator, file, scan_period)

    def _scan_from_generator(self, generator, file, scan_period):
        for image in generator:
            time_start = time.time()
            joints_uvz = self.estimator.inference_from_image(image)
            if joints_uvz is None:
                continue
            joints_xyz = self.camera.pixel_to_world(joints_uvz)
            self._save_joints_to_file(file, joints_xyz)
            self._wait_till_period(time_start, scan_period)

    def _prepare_file(self, gesture_label):
        if '_' in gesture_label:
            raise Exception("Label cannot include '_' because it is used a separator.")
        make_dir(self.subdir_path)
        timestamp = get_current_timestamp()
        timestamped_file = self.subdir_path.joinpath(F"{gesture_label}_{timestamp}.txt")
        return timestamped_file

    def _save_joints_to_file(self, file, joints):
        formatted_joints = self._format_joints(joints)
        file.write(F"{formatted_joints}\n")

    def _format_joints(self, joints):
        flattened_joints = np.reshape(joints, [-1]).astype(np.str)
        formatted_joints = ' '.join(flattened_joints)
        return formatted_joints

    def _wait_till_period(self, start_time_in_seconds, period_in_seconds):
        end_time = time.time()
        duration = end_time - start_time_in_seconds
        if duration * 1.01 < period_in_seconds:
            sleep_till_period = period_in_seconds - duration
            time.sleep(sleep_till_period)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subdir', type=str, action='store', default=None, required=True)
    parser.add_argument('--label', type=str, action='store', default=None, required=True)
    parser.add_argument('--scan-period', type=float, action='store', default=1.0)
    parser.add_argument('--camera', type=str, action='store', default='SR305')
    parser.add_argument('--plot', type=bool, action='store', default=True)
    args = parser.parse_args()

    scanner = UsecaseDatabaseScanner(args.subdir, camera=Camera(args.camera), plot_estimation=args.plot)
    scanner.scan_into_subdir(args.label, scan_period=args.scan_period)
