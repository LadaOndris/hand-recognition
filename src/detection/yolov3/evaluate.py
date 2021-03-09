import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.core.cfg.cfg_parser import Model
from src.datasets.handseg150k.dataset_bboxes import HandsegDatasetBboxes
from src.detection.yolov3 import utils
from src.detection.yolov3.dataset_generator import DatasetGenerator
from src.utils.config import TEST_YOLO_CONF_THRESHOLD, YOLO_CONFIG_FILE
from src.utils.paths import ROOT_DIR, LOGS_DIR, HANDSEG_DATASET_DIR
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score
from src.detection.yolov3.metrics import get_positives_and_negatives


def evaluate():
    model = Model.from_cfg(YOLO_CONFIG_FILE)
    model.tf_model.load_weights(LOGS_DIR.joinpath("20210112-220731/train_ckpts/ckpt_9"))

    handseg = HandsegDatasetBboxes(HANDSEG_DATASET_DIR, train_size=0.99, batch_size=model.batch_size,
                                   shuffle=False)
    test_dataset_generator = DatasetGenerator(handseg.test_batch_iterator,
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
        if batch_idx % 100 == 0:
            print(F"Evaluating batch {batch_idx}/{handseg.num_test_batches}")
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


if __name__ == '__main__':
    # disable CUDA, run on CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # y_pred, y_true = evaluate()
    # np.save('eval_y_pred.pkl', y_pred)
    # np.save('eval_y_true.pkl', y_true)

    y_pred = np.load('../../../other/eval_y_pred.pkl.npy')
    y_true = np.load('../../../other/eval_y_true.pkl.npy')

    num_thresholds = 50
    thresholds = np.linspace(0, 1, num_thresholds)
    precision = np.empty([num_thresholds])
    recall = np.empty([num_thresholds])
    for i in range(num_thresholds):
        tp, tn, fp, fn = get_positives_and_negatives(y_true, y_pred, thresholds[i])
        precision[i] = tf.math.divide_no_nan(tp, (tp + fp)).numpy()
        recall[i] = tf.math.divide_no_nan(tp, (tp + fn)).numpy()
    f1 = tf.math.divide_no_nan(2 * precision * recall, (precision + recall)).numpy()
    plt.plot(thresholds, precision)
    plt.plot(thresholds, recall)
    plt.plot(thresholds, f1)
    plt.legend(labels=['Precision', 'Recall', 'F1'])
    plt.title("Precision, Recall, and F1 metric evaluated on Tiny YOLOv3 model")
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.show()

    plt.plot(recall, precision)
    plt.title("Precision-recall curve of a trained Tiny YOLOv3 model")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

    pass
