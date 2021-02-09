from src.utils.paths import MSRAHANDGESTURE_DATASET_DIR
from src.acceptance.recognize import get_relative_distances, relative_distance_diff_single, \
    relative_distance_diff_sum
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
import glob

"""
file = MSRAHANDGESTURE_DATASET_DIR.joinpath('P0/1/000000_depth.bin')
with open(file, 'rb') as f:
    content = f.readlines()
    pass
"""


def load_joints(joints_file: str, gesture: str) -> np.ndarray:
    # joints_file = MSRAHANDGESTURE_DATASET_DIR.joinpath(F"P0/{gesture}/joint.txt")
    joints = np.genfromtxt(joints_file, skip_header=1, dtype=np.float64)
    joints = np.reshape(joints, (-1, 21, 3))
    labels = np.full(shape=(joints.shape[0]), fill_value=gesture)
    return joints, labels


def load_dataset() -> np.ndarray:
    path = MSRAHANDGESTURE_DATASET_DIR.joinpath(F"P0/")
    files = path.iterdir()
    gesture_names = []
    joints, labels = None, None

    for file in files:
        gesture_name = file.stem
        j, l = load_joints(file.joinpath('joint.txt'), gesture_name)
        if joints is None:
            joints, labels = j, l
        else:
            joints = np.concatenate((joints, j))
            labels = np.concatenate((labels, l))
        gesture_names.append(gesture_name)
    return np.unique(gesture_names), joints, labels


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.show()


def predict(joints, db_joints, db_labels):
    rds = get_relative_distances(joints, db_joints)

    masked_rds = np.where(rds < 10000, rds, np.inf)
    args_min = np.argmin(masked_rds, axis=1)

    if joints.ndim == 3:
        db_labels = np.tile(db_labels, reps=(joints.shape[0], 1))

    return np.take(db_labels, args_min)


"""
g1_joints, g1_labels = load_joints(MSRAHANDGESTURE_DATASET_DIR.joinpath(F"P0/1/joint.txt"), '1')
g2_joints, g2_labels = load_joints(MSRAHANDGESTURE_DATASET_DIR.joinpath(F"P0/2/joint.txt"), '2')
x = np.concatenate((g1_joints, g2_joints))
y = np.concatenate((g1_labels, g2_labels))
print('rd:', relative_distance_diff_single(g1_joints[0], g1_joints[1]))
print('my_rd:', relative_distance_diff_sum(g1_joints[0][np.newaxis, ...], g1_joints[1][np.newaxis, ...]))
"""

gestures, joints, labels = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(
    joints, labels, test_size=0.8, random_state=42)

y_scores = predict(X_test[:100], X_train, y_train)

accuracy = accuracy_score(y_test[:100], y_scores)
print(F"accuracy: {accuracy}")

y_scores = get_relative_distances(X_train[0], X_train)
precisions, recalls, thresholds = precision_recall_curve(
    y_train, np.squeeze(y_scores), pos_label=y_train[0], per_class=True)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
