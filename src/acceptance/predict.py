import numpy as np

from src.acceptance.base import hand_orientation, joint_relation_errors, vectors_angle


class GestureAccepter:

    def __init__(self, gesture_database_reader, jre_thresh, orientation_thresh, distance_limits=None):
        self.gesture_database = gesture_database_reader.hand_poses
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
        # The mean vector defining the center of palm joints.
        # It is used as the starting point when displaying the orientation vector.
        self.orientation_joints_mean = None
        # True if thresholds are exceeded
        self.gesture_invalid = False

    def get_jres(self):
        return self.joints_jre[:, self.predicted_gesture_idx]

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

        self.orientation, self.orientation_joints_mean = hand_orientation(keypoints)
        self.expected_orientation, _ = hand_orientation(self.predicted_gesture)
        angle_difference = np.rad2deg(vectors_angle(self.expected_orientation, self.orientation))
        self.angle_difference = self._suit_angle_for_both_hands(angle_difference)

        self.gesture_invalid = self.gesture_jre > self.jre_thresh or \
                               self.angle_difference > self.orientation_thresh

    def _suit_angle_for_both_hands(self, angle):
        """
        Do not allow angle above 90 because tt is unknown
        which hand is in the image.
        """
        if angle > 90:
            return 180 - angle
        else:
            return angle
