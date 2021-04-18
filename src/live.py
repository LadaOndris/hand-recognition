from src.acceptance import database
from src.acceptance.predict import GestureAccepter
from src.datasets.custom.dataset import CustomDataset, CustomDatasetGenerator
from src.position_estimation import HandPositionEstimator
from src.utils.camera import Camera
from src.utils.live import generate_live_images
import src.utils.plots as plots
from src.utils.paths import CUSTOM_DATASET_DIR


class GestureRecognizer:

    def __init__(self, error_thresh, plot_feedback=False):
        self.plot_feedback = plot_feedback
        self.camera = Camera('sr305')
        self.estimator = HandPositionEstimator(self.camera, cube_size=230)
        self.gesture_database = database.load_gestures()
        orientation_thres = 35
        self.gesture_accepter = GestureAccepter(self.gesture_database, error_thresh, orientation_thres)

    def start(self, image_generator):
        for image_array in image_generator:
            joints_uvz = self.estimator.inference_from_image(image_array[0])
            joints_xyz = self.camera.pixel_to_world(joints_uvz)
            self.gesture_accepter.accept_gesture(joints_xyz)
            # get the gesture label
            gesture_index = self.gesture_accepter.predicted_gesture_idx
            gesture_label = self.gesture_database.get_label(gesture_index)
            print(gesture_label)
            # plot the hand position with gesture label
            plots.plot_skeleton_with_label(joints_uvz, gesture_label)


def recognize_live():
    generator = generate_live_images()
    live_acceptance = GestureRecognizer()
    live_acceptance.start(generator)


def recognize_from_custom_dataset():
    ds = CustomDataset(CUSTOM_DATASET_DIR, batch_size=1)
    generator = CustomDatasetGenerator(ds)
    live_acceptance = GestureRecognizer(error_thresh=10000)
    live_acceptance.start(generator)


if __name__ == '__main__':
    recognize_from_custom_dataset()
