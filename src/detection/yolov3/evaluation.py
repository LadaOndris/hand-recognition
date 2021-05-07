import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve

import src.utils.plots as plots
from src.detection.yolov3.cfg.cfg_parser import Model
from src.datasets.handseg.dataset_bboxes import HandsegDatasetBboxes
from src.detection.yolov3 import utils
from src.detection.yolov3.dataset_preprocessing import DatasetPreprocessor
from src.utils.config import YOLO_CONFIG_FILE
from src.utils.paths import DOCS_DIR, HANDSEG_DATASET_DIR, LOGS_DIR


def evaluate(weights_path):
    model = Model.from_cfg(YOLO_CONFIG_FILE)
    model.tf_model.load_weights(weights_path)

    handseg = HandsegDatasetBboxes(HANDSEG_DATASET_DIR, train_size=0.99, batch_size=model.batch_size,
                                   shuffle=False, model_input_shape=model.input_shape)
    test_dataset_generator = DatasetPreprocessor(handseg.test_batch_iterator,
                                                 model.input_shape, model.yolo_output_shapes, model.anchors)

    batch_idx = 0
    y_pred = None
    y_true = None
    for batch_images, batch_y_true in test_dataset_generator:
        yolo_outputs = model.tf_model.predict(batch_images)
        batch_y_true = merge_outputs(batch_y_true)
        batch_y_pred = merge_outputs(yolo_outputs)
        if y_pred is None:
            y_pred = batch_y_pred
            y_true = batch_y_true
        else:
            y_pred = tf.concat([y_pred, batch_y_pred], axis=1)
            y_true = tf.concat([y_true, batch_y_true], axis=1)
        batch_idx += 1
        if batch_idx == handseg.num_test_batches:
            break
        if batch_idx % 10 == 0:
            print(F"Evaluating images {batch_idx}/{handseg.num_test_batches}, batch size = {model.batch_size}")
    return y_pred, y_true


def merge_outputs(y):
    y_outs = None
    for y_scale in y:
        y_scale = y_scale[..., :5]
        shape = tf.shape(y_scale)
        y_reshaped = tf.reshape(y_scale, [shape[0], shape[1] * shape[2], shape[3], shape[4]])
        if y_outs is None:
            y_outs = y_reshaped
        else:
            y_outs = tf.concat([y_outs, y_reshaped], axis=1)
    return y_outs


def generate_precision_recall_curves(y_pred_pkl, y_true_pkl, fig_scores_location=None,
                                     fig_precision_recall_location=None, show_figs=False):
    """
    Reads pkl files containing true labels and predictions and
    creates two plots. One shows precision, recall, and F1 score.
    The other one shows a Precison-Recall curve.

    Parameters
    ----------
    y_pred_pkl  : str
        A file with saved predictions to be loaded with np.load.
    y_true_pkl  : str
        A file with saved predictions to be loaded with np.load.
    fig_scores_location : str
        Target location including its file name to save the first plot.
    fig_precision_recall_location
        Target location including its file name to save the second plot.
    show_figs
        Whether to show the plots in console.
    """
    y_pred = np.load(y_pred_pkl)
    y_true = np.load(y_true_pkl)

    precision, recall, thresholds = precision_recall_curve(y_true[..., 4:5].flatten(), y_pred[..., 4:5].flatten())
    thresholds = np.concatenate([[0], thresholds])
    f1 = tf.math.divide_no_nan(2 * precision * recall, (precision + recall)).numpy()

    plots.plot_scores(x=thresholds, y=[precision, recall, f1], labels=['Precision', 'Recall', 'F1'],
                      fig_location=fig_scores_location, show_fig=show_figs)
    plots.precision_recall_curve(recall, precision, fig_location=fig_precision_recall_location, show_fig=show_figs)


def compute_iou(y_pred_pkl, y_true_pkl):
    y_pred = np.load(y_pred_pkl)
    y_true = np.load(y_true_pkl)

    pred_xywh = y_pred[..., 0:4]
    true_xywh = y_true[..., 0:4]
    true_conf = y_true[..., 4:5]
    iou_for_all_boxes = utils.tensorflow_bbox_iou(pred_xywh[:, :, :, np.newaxis, :],
                                                  true_xywh[:, :, :, np.newaxis, :])
    iou_for_true_boxes = true_conf * iou_for_all_boxes
    ious_sum = tf.reduce_sum(iou_for_true_boxes)
    nonzero_ious = tf.math.count_nonzero(iou_for_true_boxes, dtype=tf.float32)
    iou = tf.math.divide_no_nan(ious_sum, nonzero_ious)
    return iou


def evaluate_and_save(pred_file_path, true_file_path):
    # Previous Tiny YOLOv3 model that uses padding
    # weights_path = LOGS_DIR.joinpath("20210112-220731/train_ckpts/ckpt_9")
    # Tiny YOLOv3 model that uses crop
    weights_path = LOGS_DIR.joinpath("20210315-143811/train_ckpts/weights.12.h5")

    y_pred, y_true = evaluate(weights_path)
    np.save(pred_file_path, y_pred)
    np.save(true_file_path, y_true)


if __name__ == '__main__':
    pred_path = '../../../other/eval_y_pred_03-15.pkl.npy'
    true_path = '../../../other/eval_y_true_03-15.pkl.npy'
    # evaluate_and_save(pred_path, true_path)

    figure_scores_path = str(DOCS_DIR.joinpath('figures/yolo_eval_scores_03-15.pdf'))
    figure_curve_path = str(DOCS_DIR.joinpath('figures/yolo_eval_curve_03-15.pdf'))
    generate_precision_recall_curves(pred_path, true_path, show_figs=True,
                                     fig_scores_location=figure_scores_path,
                                     fig_precision_recall_location=figure_curve_path)

    # iou = compute_iou(pred_path, true_path)
    # print(iou)

    # Previous model
    # y_pred = np.load('../../../other/eval_y_pred.pkl.npy')
    # y_true = np.load('../../../other/eval_y_true.pkl.npy')

    pass
