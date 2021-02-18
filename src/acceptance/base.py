import numpy as np
import scipy.spatial


def _upper_tri(A):
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


def relative_distance_matrix(hand1: np.ndarray, hand2: np.ndarray) -> np.ndarray:
    if hand1.ndim != 3 or hand2.ndim != 3:
        raise ValueError(F"Expected dimension is 3, but is {hand1.ndim} and {hand2.ndim}")

    a = distance_matrix(hand1).astype(np.float16)
    b = distance_matrix(hand2).astype(np.float16)
    diff_matrix = a[:, np.newaxis, :, :] - b[np.newaxis, ...]
    return diff_matrix


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
    diff = relative_distance_matrix(hand1, hand2)
    return np.sum(np.abs(_upper_tri(diff)), axis=-1)


def relative_distance_diff_sum(hand1: np.ndarray, hand2: np.ndarray) -> np.ndarray:
    rd_diff = relative_distance_diff(hand1, hand2)
    return np.sum(np.abs(rd_diff), axis=-1)


def get_relative_distances(joints, db_joints):
    if joints.ndim == 2:
        return relative_distance_diff(joints[np.newaxis, ...], db_joints)
    elif joints.ndim == 3:
        return relative_distance_diff(joints, db_joints)
    else:
        raise ValueError("Bad dimension of input ndarray.")


def test_relative_distances():
    h1 = np.random.randint(0, 1000, size=(21, 3))
    h2 = np.random.randint(0, 1000, size=(21, 3))

    h1 = np.arange(21 * 3).reshape((21, 3))
    h2 = np.arange(21 * 3, 2 * 21 * 3).reshape((21, 3))

    print('rd:', relative_distance_diff_single(h1, h2))
    print('my_rd:', relative_distance_diff_sum(h1[np.newaxis, ...], h2[np.newaxis, ...]))


def average_hand_distance(joints: np.ndarray) -> np.float:
    """
    Calculates a distance from all joints to the camera
    positioned at (160, 120, 0) and averages the results.
    """
    distances = np.linalg.norm(joints - [160, 120, 0])
    return np.mean(distances)


def find_hand_rotation(joints: np.ndarray):
    """
    Determines the rotation of the hand in comparison to base position
    along each axis x, y, and z.

    First, it finds a plane going through six specific points.
    """
    A = joints
    b = scipy.linalg.norm(A, axis=1)
    return scipy.linalg.lstsq(A, -b)
    pass


def best_fitting_hyperplane(z: np.ndarray):
    """
    It approximates the best fitting hyperplane through
    these points using SVD (Singular Value Decomposition).

    Parameters
    ----------
    z   np.ndarray  A 2-D numpy array of points in space.

    Returns
    -------
    Returns a tuple. The first value returns a normal vector of the hyperplane.
    The second value is the mean value of given points.
    These values can be used to plot the normal vector at the mean coordinate
    for visualization purposes.
    """
    z_mean = np.mean(z, axis=0)
    x = z - z_mean
    u, s, vh = np.linalg.svd(x)

    # vh is a matrix containing orthonormal vectors
    # the last is a unit vector normal to the plane
    # the others form an orthonormal basis in the plane
    norm_vec = vh[-1]
    return norm_vec, z_mean


def rds_errors(hands1: np.ndarray, hands2: np.ndarray) -> np.ndarray:
    rds_diff = relative_distance_matrix(hands1, hands2)
    rds_abs = np.abs(rds_diff)
    aggregated_joint_errors = np.sum(rds_abs, axis=-1)
    averaged_joint_errors = np.divide(aggregated_joint_errors, rds_abs.shape[-1])
    return averaged_joint_errors


if __name__ == '__main__':
    # test_relative_distances()
    A = np.array([[1, 3], [2, 4], [2, 8]])
    norm_vec, mean = best_fitting_hyperplane(A)
