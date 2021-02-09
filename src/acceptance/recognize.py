import numpy as np
import scipy.spatial


def _upper_tri(A):
    # if A.ndim != 4:
    #    raise ValueError(F"Dimension should be three, but is {A.ndim}")
    r = np.arange(A.shape[A.ndim - 2])
    mask = r[:, None] < r
    return A[..., mask]


def relative_distance(hand: np.ndarray) -> np.ndarray:
    a = np.subtract.outer(hand, hand)
    return _upper_tri(a)


def relative_distance_diff_single(hand1: np.ndarray, hand2: np.ndarray) -> np.ndarray:
    a = scipy.spatial.distance_matrix(hand1, hand1)
    b = scipy.spatial.distance_matrix(hand2, hand2)
    diff = a - b
    return np.sum(np.abs(_upper_tri(diff)), axis=-1)


def distance_matrix(a):
    diff = np.empty(shape=(a.shape[0], 21, 21))

    for x in range(a.shape[0]):
        d = a[x, :, np.newaxis] - a[x, np.newaxis, :]
        diff[x, :, :] = np.sum(np.abs(d ** 2), axis=-1)
    return diff ** 0.5


def relative_distance_diff(hand1: np.ndarray, hand2: np.ndarray) -> np.ndarray:
    """
    Calculates the  difference between relative distances of two hands.
    Parameters
    ----------
    hand1 shape=(A, 21, 3)
    hand2 shape=(B, 21, 3)

    Returns
    -------
    np.ndarray of distances between all points
    Returns ndarray of shape (A * B, 210)
    """

    if hand1.ndim != 3 or hand2.ndim != 3:
        raise ValueError(F"Expected dimension is 3, but is {hand1.ndim} and {hand2.ndim}")

    a = distance_matrix(hand1).astype(np.float16)
    b = distance_matrix(hand2).astype(np.float16)
    diff = a[:, np.newaxis, :, :] - b[np.newaxis, ...]
    return np.sum(np.abs(_upper_tri(diff)), axis=-1)


def relative_distance_diff_sum(hand1: np.ndarray, hand2: np.ndarray) -> np.ndarray:
    rd_diff = relative_distance_diff(hand1, hand2)
    return np.sum(np.abs(rd_diff), axis=-1)


def get_relative_distances(joints, db_joints):
    if joints.ndim == 2:
        return relative_distance_diff(joints[np.newaxis, ...], db_joints)
        # return np.fromiter((relative_distance_diff_sum(joints, x) for x in db_joints), db_joints.dtype)
    elif joints.ndim == 3:
        return relative_distance_diff(joints, db_joints)
        # rds = np.empty(shape=(joints.shape[0], db_joints.shape[0]))
        # for idx, j in enumerate(joints):
        #    rds[idx] = [relative_distance_diff_sum(joints[idx], j) for j in db_joints]
        # return rds
    else:
        raise ValueError("Bad dimension of input ndarray.")


if __name__ == '__main__':
    h1 = np.random.randint(0, 1000, size=(21, 3))
    h2 = np.random.randint(0, 1000, size=(21, 3))

    h1 = np.arange(21 * 3).reshape((21, 3))
    h2 = np.arange(21 * 3, 2 * 21 * 3).reshape((21, 3))

    print('rd:', relative_distance_diff_single(h1, h2))
    print('my_rd:', relative_distance_diff_sum(h1[np.newaxis, ...], h2[np.newaxis, ...]))
