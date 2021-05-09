import numpy as np
import tensorflow as tf

import src.detection.yolov3.utils as utils


def get_positives_and_negatives(y_true, y_pred, conf_threshold):
    true_conf = y_true[..., 4:5]
    pred_conf = y_pred[..., 4:5]

    # convert y_pred to values 0 or 1 based on _conf_threshold
    ones = tf.ones_like(pred_conf)
    zeros = tf.zeros_like(pred_conf)
    pred_conf = tf.where(pred_conf > conf_threshold, ones, zeros)

    true_conf = tf.reshape(true_conf, [-1])
    pred_conf = tf.reshape(pred_conf, [-1])

    true_conf = tf.cast(true_conf, dtype=tf.bool)
    pred_conf = tf.cast(pred_conf, dtype=tf.bool)

    tp_vec = tf.math.logical_and(true_conf, pred_conf)
    tn_vec = tf.math.logical_not(tf.math.logical_or(true_conf, pred_conf))
    # fp = pred ^ ~true
    fp_vec = tf.math.logical_and(pred_conf, tf.math.logical_not(true_conf))

    # fn = true ^ ~pred
    fn_vec = tf.math.logical_and(tf.math.logical_not(pred_conf), true_conf)

    # tf.reduce_sum could be used, but it would require input vectors to be cast to float
    # as it cannot count booleans.
    tp = tf.math.count_nonzero(tp_vec, dtype=tf.float32)
    tn = tf.math.count_nonzero(tn_vec, dtype=tf.float32)
    fp = tf.math.count_nonzero(fp_vec, dtype=tf.float32)
    fn = tf.math.count_nonzero(fn_vec, dtype=tf.float32)

    # tf.print(tp, tn, fp, fn)
    return tp, tn, fp, fn


class YoloConfPrecisionMetric(tf.keras.metrics.Metric):

    def __init__(self, conf_threshold=.5, name='conf_precision', **kwargs):
        super(YoloConfPrecisionMetric, self).__init__(name=name, **kwargs)
        self._conf_threshold = conf_threshold
        self._total = self.add_weight(name='total', initializer='zeros')
        self._count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        precision = self._compute_precision(y_true, y_pred)
        self._total.assign_add(tf.cast(precision, dtype=tf.float32))
        self._count.assign_add(1)

    def _compute_precision(self, y_true, y_pred):
        (tp, tn, fp, fn) = get_positives_and_negatives(y_true, y_pred, self._conf_threshold)

        return tf.math.divide_no_nan(tp, (tp + fp))

    def result(self):
        return tf.math.divide_no_nan(self._total, self._count)

    def reset_states(self):
        self._total.assign(0.)
        self._count.assign(0.)


class YoloConfRecallMetric(tf.keras.metrics.Metric):

    def __init__(self, conf_threshold=.5, name='conf_recall', **kwargs):
        super(YoloConfRecallMetric, self).__init__(name=name, **kwargs)
        self._conf_threshold = conf_threshold
        self._total = self.add_weight(name='total', initializer='zeros')
        self._count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        precision = self._compute_precision(y_true, y_pred)
        self._total.assign_add(tf.cast(precision, dtype=tf.float32))
        self._count.assign_add(1)

    def _compute_precision(self, y_true, y_pred):
        (tp, tn, fp, fn) = get_positives_and_negatives(y_true, y_pred, self._conf_threshold)

        return tf.math.divide_no_nan(tp, (tp + fn))

    def result(self):
        return tf.math.divide_no_nan(self._total, self._count)

    def reset_states(self):
        self._total.assign(0.)
        self._count.assign(0.)


class YoloBoxesIoU(tf.keras.metrics.Metric):

    def __init__(self, name='boxes_iou', **kwargs):
        super(YoloBoxesIoU, self).__init__(name=name, **kwargs)
        self._total = self.add_weight(name='total', initializer='zeros')
        self._count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_boxes = y_true[..., 0:4]
        pred_boxes = y_pred[..., 0:4]

        ious = utils.tensorflow_bbox_iou(true_boxes[..., np.newaxis, :],
                                         pred_boxes[..., np.newaxis, :])

        # Take into account only IoU of true boxes
        true_conf = y_true[..., 4:5]
        ious_masked = tf.boolean_mask(ious, true_conf)

        # If there is no true box in the batch, tf.reduce_mean returns nan
        # and spoils the whole metric
        if tf.size(ious_masked) > 0:
            # Compute mean value of all those IoUs
            mean_iou = tf.reduce_mean(ious_masked)

            self._total.assign_add(mean_iou)
            self._count.assign_add(1)

    def result(self):
        return tf.math.divide_no_nan(self._total, self._count)

    def reset_states(self):
        self._total.assign(0.)
        self._count.assign(0.)
