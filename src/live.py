from src.acceptance import database
from src.acceptance.predict import GestureAccepter
from src.position_estimation import HandPositionEstimator
from src.utils.camera import Camera
from src.utils.live import generate_live_images


class LiveGestureRecognizer:

    def __init__(self, error_thresh, plot_feedback=False):
        self.plot_feedback = plot_feedback
        camera = Camera('sr305')
        self.estimator = HandPositionEstimator(camera, cube_size=230)
        self.live_generator = generate_live_images()
        self.gesture_database = database.load_gestures()
        orientation_thres = 35
        self.gesture_accepter = GestureAccepter(self.gesture_database, error_thresh, orientation_thres)

    def start(self):
        for image_array in self.live_generator:
            joints_xyz = self.estimator.inference_from_image(image_array, return_xyz=True)
            self.gesture_accepter.accept_gesture(joints_xyz)
            # get the gesture label

            # plot the hand position with gesture label



if __name__ == '__main__':
    live_acceptance = LiveGestureRecognizer()
    live_acceptance.start()
