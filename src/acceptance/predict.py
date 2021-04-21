import numpy as np
from src.acceptance.base import joint_relation_errors, hand_orientation, vectors_angle


class GestureAccepter:

    def __init__(self, gesture_database, jre_thresh, orientation_thresh, distance_limits=None):
        self.gesture_database = gesture_database
        self.jre_thresh = jre_thresh
        self.orientation_thresh = orientation_thresh
        self.distance_limits = distance_limits

        # Statistics created for each prediction
        # --------------------------------------
        # The joint relation errors between given hand pose and the whole database
        self.joints_jre = None
        # Index of the gesture from the dataset that is most similar to the given
        self.predicted_gesture_idx = None
        # The gesture from the dataset that is most similar to the given
        self.predicted_gesture = None
        # Aggregated (summed) joint relation errors between the given hand pose
        # and the most similar one from the dataset
        self.gesture_jre = None
        # The norm vector defining the orientation of the given gesture
        self.orientation = None
        # The norm vector defining the orientation of the most similar hand pose
        self.expected_orientation = None
        # The orientation difference between given and most similar poses
        # represented as an angle
        self.angle_difference = None
        # True if thresholds are exceeded
        self.gesture_invalid = False

    def accept_gesture(self, keypoints: np.ndarray):
        """
        SHAPE??????

        Predicts a gesture for the given keypoints.
        Compares given keypoints to the ones stored in the database.

        Parameters
        ----------
        keypoints ndarray of 21 keypoints, shape (batch_size, joints, coords)
        """
        # files, keypoints_db = database.load_gestures()
        self.joints_jre = joint_relation_errors(keypoints, self.gesture_database)
        aggregated_errors = np.sum(self.joints_jre, axis=-1)
        self.predicted_gesture_idx = np.argmin(aggregated_errors, axis=-1)
        self.predicted_gesture = self.gesture_database[self.predicted_gesture_idx, ...]
        self.gesture_jre = aggregated_errors[..., self.predicted_gesture_idx]

        self.orientation, _ = hand_orientation(keypoints)
        self.expected_orientation, _ = hand_orientation(self.predicted_gesture)
        self.angle_difference = np.rad2deg(vectors_angle(self.expected_orientation, self.orientation))

        self.gesture_invalid = self.gesture_jre > self.jre_thresh or \
                               self.angle_difference > self.orientation_thresh
