import numpy as np
import scipy.spatial


def _upper_tri(A):
    """
    Returns an upper triangle of a given matrix.
    """
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
        # Subtract x,y,z coordinates
        d = a[x, :, np.newaxis] - a[x, np.newaxis, :]
        # Compute Euclidean distance using Pythagoras
        diff[x, :, :] = np.sum(np.abs(d ** 2), axis=-1)
    return diff ** 0.5


def relative_distance_matrix(hand1: np.ndarray, hand2: np.ndarray) -> np.ndarray:
    if hand1.ndim != 3 or hand2.ndim != 3:
        raise ValueError(F"Expected dimension is 3, but is {hand1.ndim} and {hand2.ndim}")

    a = distance_matrix(hand1).astype(np.float16)
    b = distance_matrix(hand2).astype(np.float16)
    a *= get_scale_factors(a)[:, np.newaxis, np.newaxis]
    b *= get_scale_factors(b)[:, np.newaxis, np.newaxis]
    diff_matrix = a[:, np.newaxis, :, :] - b[np.newaxis, ...]
    return diff_matrix


def get_scale_factors(distance_matrix, standard_finger_length=65.):
    """
    Computes factors for scaling hand based on a mean
    finger length. The actual mean lengths are divided
    by a standard length of 65 mm.

    Parameters
    ----------
    distance_matrix
    standard_finger_length
        A standard mean finger length

    Returns
    -------
    scale_factors
    """
    if distance_matrix.ndim != 3:
        raise Exception(F"Invalid rank of distance matrix, expected 3.")
    fingers_lengths = fingers_length_from_distance_matrix(distance_matrix)
    mean_lengths = np.mean(fingers_lengths, axis=1)
    scale_factors = standard_finger_length / mean_lengths
    return scale_factors


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


"""
def relative_distance_diff_sum(hand1: np.ndarray, hand2: np.ndarray) -> np.ndarray:
    rd_diff = relative_distance_diff(hand1, hand2)
    return np.sum(np.abs(rd_diff), axis=-1)
"""


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
    print('my_rd:', relative_distance_diff(h1[np.newaxis, ...], h2[np.newaxis, ...]))


def hand_distance(joints: np.ndarray, camera_position=[160, 120, 0]) -> np.float:
    """
    Calculates a distance from camera to the hand
    by computing the distance of each joint and averaging the distances.

    Returns 1-D scalar
    -------
    """
    distances = np.linalg.norm(joints - camera_position)
    return np.mean(distances)


def hand_rotation(joints: np.ndarray):
    """
    Determines the rotation of the hand in comparison to base position
    along each axis x, y, and z.

    It finds a plane going through six specific points of a hand,
    and returns the normal vector of the plane and a mean value of the six points.
    """

    # 0: wrist,
    # 1-4: index_mcp, index_pip, index_dip, index_tip,
    # 5-8: middle_mcp, middle_pip, middle_dip, middle_tip,
    # 9-12: ring_mcp, ring_pip, ring_dip, ring_tip,
    # 13-16: little_mcp, little_pip, little_dip, little_tip,
    # 17-20: thumb_mcp, thumb_pip, thumb_dip, thumb_tip

    palm_joints = np.take(joints, [0, 1, 5, 9, 13], axis=0)
    norm_vec, mean = best_fitting_hyperplane(palm_joints)
    return norm_vec, mean


def best_fitting_hyperplane(z: np.ndarray):
    """
    It approximates the best fitting hyperplane through
    these points using SVD (Singular Value Decomposition).

    Parameters
    ----------
    z   np.ndarray
        A 2-D numpy array of points in space.
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


def joint_relation_errors(hands1: np.ndarray, hands2: np.ndarray, relative_distance_matrix=None) -> np.ndarray:
    """
    Computes the average relative difference of joint positions of one hand
    in comparison to the second hand.
    First, it computes relative distances for each hand. Relative distances are represented
    in a matrix with shape (21, 21), as the relative distance is computed between each joint.
    Then, it subtracts these relative distances producing a so-called Relative Distance Matrix.
    This operation produces the following shapes: (21, 21) - (21, 21) = (21, 21).
    Relative distances for each joint are summed and averaged by the number of joints, producing
    21 scalars, which are the actual joint relation errors.

    Returns (21,) np.ndarray Vector
        A vector of 21 scalars: an error for each joint.
    -------

    """
    if relative_distance_matrix is None:
        relative_distances = relative_distance_matrix(hands1, hands2)
    distances_abs = np.abs(relative_distances)
    aggregated_joint_errors = np.sum(distances_abs, axis=-1)
    joints_count = distances_abs.shape[-1] - 1  # 21 - 1 = 20, do not count the joint itself
    averaged_joint_errors = np.divide(aggregated_joint_errors, joints_count)
    return averaged_joint_errors

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def vectors_angle(v1, v2):
    """
    Returns the angle between two vectors.
    """
    v1 = unit_vector(v1)
    v2 = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


def fingers_length(joints):
    """
    Returns length of each finger by summing the
    distance between MCP, PIP, DIP, and TIP joints in 3D coordinates.

    Parameters
    ----------
    joints : (batch_size, 21 joints, 3)
        0-th joint is the wrist,
        1--4 Index (MCP, PIP, DIP, TIP)
        5--8 Middle (MCP, PIP, DIP, TIP)
        9--12 Ring (MCP, PIP, DIP, TIP)
        13--16 Little (MCP, PIP, DIP, TIP)
        17--20 Thumb (MCP, PIP, DIP, TIP)

    Returns
    -------
    shape (batch_size, 5)
    an array of 5 scalars for each hand
    """
    distances = distance_matrix(joints)
    finger_lengths = fingers_length_from_distance_matrix(distances)
    return finger_lengths


def fingers_length_from_distance_matrix(distance_matrix):
    """
    Returns length of each finger by summing the
    distance between MCP, PIP, DIP, and TIP joints in 3D coordinates.

    Parameters
    ----------
    distance_matrix : shape [batch_size, 21, 21]
    Matrix of distances between each joint.

    Returns
    -------
    shape (batch_size, 5)
    an array of 5 scalars for each hand
    """
    # Gather distances between the MCP, DIP, PIP, and tip fingers
    # of each finger, producing shape [batch_size, 15]
    x = np.array([1, 2, 3,
                  5, 6, 7,
                  9, 10, 11,
                  13, 14, 15,
                  17, 18, 19])
    y = x + 1
    joint_distances = distance_matrix[:, x, y]

    # Reshape to [batch_size, 5 fingers, 3 distances]
    finger_distances = np.reshape(joint_distances, [-1, 5, 3])
    finger_lengths = np.sum(finger_distances, axis=2)
    return finger_lengths


if __name__ == '__main__':
    # test_relative_distances()
    A = np.array([[1, 3], [2, 4], [2, 8]])
    norm_vec, mean = best_fitting_hyperplane(A)
