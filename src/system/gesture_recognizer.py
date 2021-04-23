from src.system import database_reader
from src.acceptance.predict import GestureAccepter
from src.datasets.custom.dataset import CustomDataset, CustomDatasetGenerator
from src.system.database_reader import UsecaseDatabaseReader
from src.system.hand_position_estimator import HandPositionEstimator
from src.utils.camera import Camera
from src.utils.live import generate_live_images
import src.utils.plots as plots
from src.utils.paths import CUSTOM_DATASET_DIR


class GestureRecognizer:

    def __init__(self, error_thresh, database_subdir, orientation_thresh, plot_feedback=False):
        self.plot_feedback = plot_feedback
        self.camera = Camera('sr305')
        self.estimator = HandPositionEstimator(self.camera, cube_size=230)
        self.database_reader = UsecaseDatabaseReader()
        self.database_reader.load_from_subdir(database_subdir)
        self.gesture_accepter = GestureAccepter(self.database_reader, error_thresh, orientation_thresh)

    def start(self, image_generator):
        for image_array in image_generator:
            joints_uvz = self.estimator.inference_from_image(image_array[0])
            # Detection failed, continue to next image
            if joints_uvz is None:
                continue
            joints_xyz = self.camera.pixel_to_world(joints_uvz)
            self.gesture_accepter.accept_gesture(joints_xyz)
            if self.gesture_accepter.gesture_invalid:
                gesture_label = 'None'
                print(F"JRE: {self.gesture_accepter.gesture_jre}, "
                      F"Orient. diff: {self.gesture_accepter.angle_difference}")
            else:
                # get the gesture label
                gesture_index = self.gesture_accepter.predicted_gesture_idx
                gesture_label = self.database_reader.get_label_by_index(gesture_index)
                print(F"Label: {gesture_label}, "
                      F"JRE: {self.gesture_accepter.gesture_jre}, "
                      F"Orient. diff: {self.gesture_accepter.angle_difference}")
            # plot the hand position with gesture label
            image_subregion = self.estimator.get_cropped_image()
            joints_subregion = self.estimator.convert_to_cropped_coords(joints_uvz)
            plots.plot_skeleton_with_label(image_subregion, joints_subregion, gesture_label)

def recognize_live():
    generator = generate_live_images()
    live_acceptance = GestureRecognizer(error_thresh=10000, orientation_thresh=60, database_subdir='test')
    live_acceptance.start(generator)


def recognize_from_custom_dataset():
    ds = CustomDataset(CUSTOM_DATASET_DIR, batch_size=1)
    generator = CustomDatasetGenerator(ds)
    live_acceptance = GestureRecognizer(error_thresh=200, orientation_thresh=100, database_subdir='test')
    live_acceptance.start(generator)


if __name__ == '__main__':
    recognize_from_custom_dataset()
