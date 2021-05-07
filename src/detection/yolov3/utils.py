import numpy as np
import tensorflow as tf

depth_image_cmap = 'gist_yarg'
prediction_box_color = '#B73229'
blue_color = '#293e65'
boxes_color = '#141d32'


def bbox_iou(boxes1, boxes2):
    """
    boxes1.shape (n, 4)
    boxes2.shape (m, 4)

    Returns
        Returns an array of iou for each combination of possible intersection.
    """
    # convert to numpy arrays
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]  # width * height
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]  # width * height

    # Convert xywh to x1,y1,x2,y2 (top left and bottom right point).
    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # Find the length of x and y where the rectangles overlap.
    # If the length is less than 0, they do not overlap.
    intersection_lengths = np.maximum(right_down - left_up, 0.0)
    intersection_area = intersection_lengths[..., 0] * intersection_lengths[..., 1]
    union_area = boxes1_area + boxes2_area - intersection_area

    return np.nan_to_num(intersection_area / union_area)


def tensorflow_bbox_iou(boxes1, boxes2):
    boxes1 = tf.cast(boxes1, dtype=tf.float32)
    boxes2 = tf.cast(boxes2, dtype=tf.float32)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1_xy, boxes1_wh = tf.split(boxes1, [2, 2], axis=-1)
    boxes2_xy, boxes2_wh = tf.split(boxes2, [2, 2], axis=-1)

    boxes1 = tf.concat([boxes1_xy - boxes1_wh * 0.5,
                        boxes1_xy + boxes1_wh * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2_xy - boxes2_wh * 0.5,
                        boxes2_xy + boxes2_wh * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = 1.0 * tf.math.divide_no_nan(inter_area, union_area)

    return tf.where(tf.math.is_nan(iou), tf.zeros_like(iou), iou)


def non_max_suppression(inputs, model_size, max_output_size,
                        max_output_size_per_class, iou_threshold, confidence_threshold):
    """
    

    Parameters
    ----------
    inputs : TYPE
        A 3-D Tensor of shape [batch_size, num_boxes, 5 + irrelevant]
    model_size : TYPE
        DESCRIPTION.
    max_output_size : TYPE
        DESCRIPTION.
    max_output_size_per_class : TYPE
        DESCRIPTION.
    iou_threshold : TYPE
        DESCRIPTION.
    confidence_threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    boxes : TYPE
        A 3-D Tensor of shape [batch_size, valid_boxes, 4].
    scores : TYPE
        DESCRIPTION.
    num_valid_detections : TYPE
        A 1-D Tensor of shape [batch_size].

    """
    bboxes, confs, other = tf.split(inputs, [4, 1, -1], axis=-1)
    bboxes = tf.expand_dims(bboxes, axis=2)  # the third dimension is 1 according to docs

    boxes, scores, classes, num_valid_boxes = tf.image.combined_non_max_suppression(
        boxes=bboxes,
        scores=confs,
        max_output_size_per_class=max_output_size_per_class,
        max_total_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=confidence_threshold,
        clip_boxes=False
    )

    return boxes, scores, num_valid_boxes


def output_boxes(inputs, model_size, max_output_size, max_output_size_per_class,
                 iou_threshold, confidence_threshold):
    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    top_left_x = center_x - width * 0.5
    top_left_y = center_y - height * 0.5
    bottom_right_x = center_x + width * 0.5
    bottom_right_y = center_y + height * 0.5

    inputs = tf.concat([top_left_x,
                        top_left_y,
                        bottom_right_x,
                        bottom_right_y,
                        confidence, classes], axis=-1)

    boxes_dicts = non_max_suppression(inputs, model_size, max_output_size,
                                      max_output_size_per_class, iou_threshold, confidence_threshold)

    return boxes_dicts


def boxes_from_yolo_outputs(yolo_outputs, batch_size, model_size, conf_thresh, max_boxes=2, iou_thresh=.5):
    scale1_outputs = tf.reshape(yolo_outputs[0], [batch_size, -1, 6])
    scale2_outputs = tf.reshape(yolo_outputs[1], [batch_size, -1, 6])
    predictions_for_the_image = tf.concat([scale1_outputs, scale2_outputs], axis=1)  # outputs for the whole batch

    boxes, scores, nums = output_boxes(predictions_for_the_image, model_size, max_boxes, max_boxes, iou_thresh,
                                       conf_thresh)
    return boxes, scores, nums
