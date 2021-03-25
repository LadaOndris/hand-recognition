from src.acceptance.base import get_relative_distances
import src.datasets.MSRAHandGesture.dataset as msra_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
import sklearn


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.show()


def predict(joints, db_joints, db_labels):
    rds = get_relative_distances(joints, db_joints)

    # rds = np.where(rds < 10000, rds, np.inf)
    args_min = np.argmin(rds, axis=1)

    if joints.ndim == 3:
        db_labels = np.tile(db_labels, reps=(joints.shape[0], 1))

    return np.take(db_labels, args_min)


def sample_train_test_split(joints, labels, class_names, indices, train_samples_per_class):
    x_train = x_test = y_train = y_test = None
    for i, class_name in enumerate(class_names):
        class_indices = indices == i
        class_joints = joints[class_indices]
        class_labels = labels[class_indices]
        if x_train is None:
            x_train = class_joints[:train_samples_per_class]
            x_test = class_joints[train_samples_per_class:]
            y_train = class_labels[:train_samples_per_class]
            y_test = class_labels[train_samples_per_class:]
        else:
            x_train = np.concatenate([x_train, class_joints[:train_samples_per_class]])
            x_test = np.concatenate([x_test, class_joints[train_samples_per_class:]])
            y_train = np.concatenate([y_train, class_labels[:train_samples_per_class]])
            y_test = np.concatenate([y_test, class_labels[train_samples_per_class:]])
    return x_train, x_test, y_train, y_test


def false_prediction_pairs(y_pred, y_true):
    cond = y_pred != y_true
    false_preds = np.stack([y_true[cond], y_pred[cond]], axis=-1)
    false_pairs, counts = np.unique(false_preds, axis=0, return_counts=True)
    return false_pairs, counts


def predict_for_different_train_sizes(repetitions=10):
    gestures, joints, labels = msra_dataset.load_dataset()
    unique_labels, indices = np.unique(labels, return_inverse=True)
    print(F"The accuracies are averaged over {repetitions} runs.")
    for i in range(1, 11):
        accuracies = []
        for rep in range(repetitions):
            # Shuffle so that the sampling is random
            joints, labels = sklearn.utils.shuffle(joints, labels)
            x_train, x_test, y_train, y_test = sample_train_test_split(
                joints, labels, unique_labels, indices, train_samples_per_class=i)
            # Shuffle again the test set beceuase the sampling put the same
            # class labels next to each other
            x_test, y_test = sklearn.utils.shuffle(x_test, y_test)
            test_num = 10000
            y_pred = predict(x_test[:test_num], db_joints=x_train, db_labels=y_train)
            accuracy = accuracy_score(y_test[:test_num], y_pred)
            accuracies.append(accuracy)
        print(F"Accuracy (samples per class = {i}): {np.mean(accuracies)}")
    print("The following pairs were predicted incorrectly (true, pred):")
    pairs, counts = false_prediction_pairs(y_pred, y_test[:test_num])
    print(zip(counts, pairs))


def plot_predition():
    """
    The accuracy fromthe randomnly sampled trained dataset works.
    The plot who knows..
    """
    gestures, joints, labels = msra_dataset.load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        joints, labels, test_size=0.999, random_state=42)

    y_scores = predict(X_test[:1000], X_train, y_train)

    accuracy = accuracy_score(y_test[:1000], y_scores)
    print(F"accuracy: {accuracy}")

    y_scores = get_relative_distances(X_train[0], X_train)
    precisions, recalls, thresholds = precision_recall_curve(
        np.where(y_train == y_train[0], y_train[0], '-'), np.squeeze(y_scores), pos_label=y_train[0])
    # recall_score(np.where(y_train == y_train[0], y_train[0], '-'),
    # np.where(np.squeeze(y_scores) < 2200, y_train[0], '-'), pos_label=y_train[0])
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)


# plot_predition()
predict_for_different_train_sizes()
