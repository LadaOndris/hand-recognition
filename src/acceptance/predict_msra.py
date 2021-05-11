from src.acceptance.base import get_relative_distances
import src.datasets.msra.dataset as msra_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import sklearn
import src.utils.plots as plots
from src.utils.paths import DOCS_DIR


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.show()


def predict(joints, db_joints, db_labels):
    """
    Gets Joint Relation Errors for each pair of hands---
    betweeen the joints and db_joints---and
    classifies the gesture as the one with the lowest error.

    Parameters
    ----------
    joints : shape [batch_size, 21]
        A batch of input hand poses.
    db_joints : shappe [db_size, 21]
        The database of hand poses (train dataset).
    db_labels : shape [db_size]
        The gesture labels for the database of hand poses.

    Returns
    -------
    Gesture class labels of each sample in the batch.
    """
    rds = get_relative_distances(joints, db_joints)

    # rds = np.where(rds < 10000, rds, np.inf)
    args_min = np.argmin(rds, axis=1)

    if joints.ndim == 3:
        db_labels = np.tile(db_labels, reps=(joints.shape[0], 1))

    return np.take(db_labels, args_min)


def sample_train_test_split(joints, labels, class_names, indices, train_samples_per_class):
    """
    Randomly samples train dataset and the rest puts
    in the test dataset. The number of training samples is specified
    by the train_samples_per_class argument.

    Parameters
    ----------
    joints
    labels
    class_names
    indices
    train_samples_per_class : int
        Number of train samples to select per each class.

    Returns
    -------
    dataset : Tuple(x_train, x_test, y_train, y_test)
    """

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
    """
    Prints pairs of predicted and true classes that differ.

    Returns
    -------
    false_pairs
        The pairs of classes that differ.
    counts
        Number of occurences of the pairs.
    """
    cond = y_pred != y_true
    false_preds = np.stack([y_true[cond], y_pred[cond]], axis=-1)
    false_pairs, counts = np.unique(false_preds, axis=0, return_counts=True)
    return false_pairs, counts


def predict_for_different_train_sizes(repetitions=10):
    """
    Performs gesture recognition on MSRA dataset.
    Randomly samples N frames for gesture, creating the training dataset.
    Then, the gesture recognition is evaluated on the rest of the dataset.
    All accuracies are printed.

    Parameters
    ----------
    repetitions : int
    Number of times to repeat the experiment.

    Returns
    -------
    None
    """
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
    Prints accuracy score of the gesture recognition on the msra dataset,
    and plots precision recall curve.

    The accuracy from the randomnly sampled trained dataset works.
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


def plot_accuracy_vs_train_sizes():
    """
    Plots the accuracy score produced by the
    predict_for_different_train_sizes function.
    """
    fig_location = DOCS_DIR.joinpath('figures/evaluation/gesture_recognition_evaluation_msra.pdf')
    acc_without_length_norm = [0.47216, 0.68169, 0.79358, 0.83532, 0.86857,
                               0.90493, 0.91702, 0.94044, 0.94254, 0.94857]

    acc_with_length_norm = [0.62158, 0.77591, 0.88974, 0.91951, 0.94561,
                            0.94694, 0.96007, 0.96839, 0.96998, 0.97280]
    samples_per_class = np.arange(1, 11)

    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({"font.size": 24})
    # sns.set_style("whitegrid", {'axes.grid': False})
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(samples_per_class, acc_without_length_norm)
    ax.plot(samples_per_class, acc_with_length_norm)
    ax.set_ylim([0, 1])
    ax.set_xlim([1, 10])
    ax.set_xlabel('Number of samples per class', labelpad=20)
    ax.set_ylabel('Accuracy', labelpad=20)
    ax.tick_params(axis='x', pad=15)
    ax.tick_params(axis='y', pad=15)
    fig.tight_layout()
    plots.save_show_fig(fig, fig_location, True)


if __name__ == '__main__':
    # predict_for_different_train_sizes()
    plot_accuracy_vs_train_sizes()
