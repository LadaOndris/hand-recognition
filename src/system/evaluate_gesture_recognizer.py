import numpy as np
from sklearn.metrics import precision_recall_curve, det_curve

import src.utils.plots as plots
from src.datasets.custom.dataset import CustomDataset
from src.system.gesture_recognizer import GestureRecognizer
from src.utils.paths import CUSTOM_DATASET_DIR, OTHER_DIR


def save_produced_metrics_on_custom_dataset():
    ds = CustomDataset(CUSTOM_DATASET_DIR, batch_size=8, right_hand_only=True)
    live_acceptance = GestureRecognizer(error_thresh=200, orientation_thresh=50, database_subdir='test')
    jres, angles, pred_labels, true_labels = live_acceptance.produce_jres(ds)
    custom_dataset_jre_path = OTHER_DIR.joinpath('custom_dataset_jres_right_hand.npz')
    np.savez(custom_dataset_jre_path, jres=jres, angles=angles, pred_labels=pred_labels, true_labels=true_labels)


def evaluation_plots_from_saved_metrics():
    data = load_saved_metrics()
    jres = data['jres']
    true_labels = data['true_labels']

    true_labels[true_labels != '1'] = '0'

    fpr, fnr, thresholds = det_curve_threshold_based(true_labels, jres, pos_label='1')
    plots.plot_scores(thresholds, y=[fpr, fnr], labels=['False positive rate', ' False negative rate'])

    precision, recall, thresholds = precision_recall_curve_threshold_based(true_labels, jres, pos_label='1')
    plots.plot_scores(thresholds, y=[precision, recall], labels=['Precision', 'Recall'])


def jre_histogram_from_saved_metrics():
    data = load_saved_metrics()
    jres = data['jres']
    orient_diffs = data['angles']
    true_labels = data['true_labels']
    true_jres = jres[true_labels == '1']
    true_diffs = orient_diffs[true_labels == '1']
    plots.histogram(true_jres, label='Joint Relation Error (Gesture 1)', range=(0, 500))
    plots.histogram(true_diffs, label='Orientation difference [degrees]', range=(0, 90))

    none_gesture_jres = jres[true_labels == '0']
    none_gesture_diffs = orient_diffs[true_labels == '0']
    plots.histogram(none_gesture_jres, label='Joint Relation Error (No gesture)', range=(0, 500))
    plots.histogram(none_gesture_diffs, label='Orientation difference [degrees]', range=(0, 90))


def load_saved_metrics():
    custom_dataset_jre_path = OTHER_DIR.joinpath('custom_dataset_jres_right_hand.npz')
    data = np.load(custom_dataset_jre_path, allow_pickle=True)
    return data

def det_curve_threshold_based(y_true, y_pred, pos_label=1):
    true = y_true.ravel()
    pred = y_pred.ravel()
    pred_max = np.max(pred)
    pred_inversed = pred_max - pred
    fpr, fnr, thresholds = det_curve(true, pred_inversed, pos_label='1')
    thresholds = pred_max - thresholds
    return fpr, fnr, thresholds

def precision_recall_curve_threshold_based(y_true, y_pred, pos_label=1):
    """
    Computes precision and recall for all possible thresholds.
    The y_pred value is expected not to be a probability score.
    """
    # true_labels = np.array([1,1, 1, 0, 0, 0])
    # jres = np.array([80, 124, 148, 150, 156, 220])
    true = y_true.ravel()
    pred = y_pred.ravel()
    pred_max = np.max(pred)
    pred_inversed = pred_max - pred
    precision, recall, thresholds = precision_recall_curve(true, pred_inversed, pos_label=pos_label)
    thresholds = np.concatenate([pred_max - thresholds])
    return precision[:-1], recall[:-1], thresholds


if __name__ == '__main__':
    import time

    start = time.time()
    evaluation_plots_from_saved_metrics()
    jre_histogram_from_saved_metrics()
    # save_produced_metrics_on_custom_dataset()
    end = time.time()
    print("It took for 125 batches x 8 images:", end - start)
