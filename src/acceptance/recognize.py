import numpy as np


def _upper_tri(A):
    r = np.arange(A.shape[0])
    mask = r[:, None] < r
    return A[mask]


def calc_relative_distance(hand: np.ndarray) -> np.ndarray:
    a = np.subtract.outer(hand, hand)
    return _upper_tri(a)


def calc_relative_distance_diff(hand1: np.ndarray, hand2: np.ndarray) -> np.ndarray:
    """
    Calculates the  difference between relative distances of two hands.
    Parameters
    ----------
    hand1 21 keypoints (3D locations) of a hand
    hand2 21 keypoints (3D locations) of another hand

    Returns
    -------
    np.ndarray of distances between all points
    """

    a = np.subtract.outer(hand1, hand1)
    b = np.subtract.outer(hand2, hand2)
    return _upper_tri(a - b)


def calc_relative_distance_diff_abs(hand1: np.ndarray, hand2: np.ndarray) -> np.ndarray:
    rd_diff = calc_relative_distance_diff(hand1, hand2)
    return np.abs(rd_diff)


if __name__ == '__main__':
    hand1 = np.random.randint(0, 1000, size=21)
    hand2 = np.random.randint(0, 1000, size=21)
    print(calc_relative_distance_diff(hand1, hand2))
    print(calc_relative_distance_diff_abs(hand1, hand2))
