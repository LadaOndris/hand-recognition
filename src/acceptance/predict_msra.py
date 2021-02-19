from src.acceptance.base import get_relative_distances, relative_distance_diff_single, \
    relative_distance_diff_sum
import src.datasets.MSRAHandGesture.dataset as msra_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np


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

gestures, joints, labels = msra_dataset.load_dataset()
X_train, X_test, y_train, y_test = train_test_split(
    joints, labels, test_size=0.8, random_state=42)

y_scores = predict(X_test[:100], X_train, y_train)

accuracy = accuracy_score(y_test[:100], y_scores)
print(F"accuracy: {accuracy}")

y_scores = get_relative_distances(X_train[0], X_train)
precisions, recalls, thresholds = precision_recall_curve(
    np.where(y_train == y_train[0], y_train[0], '-'), np.squeeze(y_scores), pos_label=y_train[0])
# recall_score(np.where(y_train == y_train[0], y_train[0], '-'),
# np.where(np.squeeze(y_scores) < 2200, y_train[0], '-'), pos_label=y_train[0])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
