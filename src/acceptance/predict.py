import numpy as np
from src.acceptance import database
from src.acceptance.base import joint_relation_errors, hand_distance, hand_rotation, get_relative_distances, vectors_angle


def predict_gesture(keypoints: np.ndarray, hands_db):
    """
    Finds the most likely gesture from hands database.

    Parameters
    ----------
    keypoints  np.ndarray of 21 keypoints, shape (21, 3)
    hands_db

    Returns Tuple (gesture_index, RD diff, orientation diff, distance)
    -------
    """
    rds = get_relative_distances(keypoints, hands_db)
    min_rds_idx = np.argmin(rds)
    predicted_gesture = hands_db[min_rds_idx]

    given_hand_orientation, _ = hand_rotation(keypoints)
    predicted_hand_orientation, _ = hand_rotation(predicted_gesture)
    orientation_diff = vectors_angle(given_hand_orientation, predicted_hand_orientation)

    distance = hand_distance(keypoints)

    return min_rds_idx, rds[min_rds_idx], orientation_diff, distance


def accept_gesture(keypoints: np.ndarray, rd_threshold, orientation_threshold, distance_limits):
    """
    Predicts a gesture for the given keypoints.
    Compares given keypoints to the ones stored in the database.

    Parameters
    ----------
    keypoints
    rd_threshold
    orientation_threshold
    distance_limits

    Returns Tuple (gesture_idx, rd errors, orientation error)
    -------
    """
    files, keypoints_db = database.load_gestures()
    gesture_idx, rd_diff, orientation_diff, distance = predict_gesture(keypoints, keypoints_db)

    most_similar_gesture = keypoints_db[gesture_idx]
    rd_errors = joint_relation_errors(keypoints, most_similar_gesture)
    orientation_error = orientation_threshold - orientation_diff

    if rd_diff > rd_threshold or orientation_diff > orientation_threshold:
        gesture_idx = None

    return gesture_idx, rd_errors, orientation_error
