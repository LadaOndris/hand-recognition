import numpy as np
from src.acceptance.base import joint_relation_errors, hand_rotation, vectors_angle


class GestureAccepter:

    def __init__(self, gesture_database, jre_thresh, orientation_thresh, distance_limits=None):
        self.gesture_database = gesture_database
        self.jre_thres = jre_thresh
        self.orientation_thresh = orientation_thresh
        self.distance_limits = distance_limits

        # Statistics created for each prediction
        # The joint relation errors between given hand pose and the whole database
        self.joints_errors = None
        # Index of the gesture from the dataset that is most similar to the given
        self.predicted_gesture_idx = None
        # The gesture from the dataset that is most similar to the given
        self.predicted_gesture = None
        # Aggregated (summed) joint relation errors between the given hand pose
        # and the most similar one from the dataset
        self.gesture_jre = None
        # The angle of the given gesture
        self.angle = None
        # The angle of the most similar hand pose
        self.expected_angle = None
        # The orientation difference between given and most similar poses
        # represented as an angle
        self.angle_difference = None

    def accept_gesture(self, keypoints: np.ndarray):
        """
        Predicts a gesture for the given keypoints.
        Compares given keypoints to the ones stored in the database.

        Parameters
        ----------
        keypoints np.ndarray of 21 keypoints, shape (21, 3)
        """
        # files, keypoints_db = database.load_gestures()
        self.joints_errors = joint_relation_errors(keypoints, self.gesture_database)
        aggregated_errors = np.sum(self.joints_errors, axis=-1)
        self.predicted_gesture_idx = np.argmin(aggregated_errors)
        self.predicted_gesture = self.gesture_database[self.predicted_gesture_idx]
        self.gesture_jre = aggregated_errors[self.predicted_gesture_idx]

        self.angle, _ = hand_rotation(keypoints)
        self.expected_angle, _ = hand_rotation(self.predicted_gesture)
        self.angle_difference = vectors_angle(self.expected_angle, self.angle)

        if self.gesture_jre > self.jre_thresh or self.angle_difference > self.orientation_thresh:
            gesture_idx = None

        return gesture_idx
