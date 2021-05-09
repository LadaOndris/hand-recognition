class GestureAcceptanceResult:

    def __init__(self):
        # Statistics created for each prediction
        # --------------------------------------
        # The joint relation errors between given hand pose and the whole database
        self.joints_jre = None
        # Index of the gesture from the dataset that is most similar to the given
        self.predicted_gesture_idx = None
        # The gesture from the dataset that is most similar to the given
        self.predicted_gesture = None
        # The predicted gesture label
        self.gesture_label = None
        # The expected gesture label
        self.expected_gesture_label = None
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
        self.is_gesture_valid = False
