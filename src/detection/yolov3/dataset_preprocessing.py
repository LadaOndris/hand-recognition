import numpy as np
import tensorflow as tf

from src.detection.yolov3.utils import bbox_iou

"""
Preprocesses bounding boxes from tf.Dataset and produces y_true.
"""


class DatasetPreprocessor:

    def __init__(self, dataset_bboxes_iterator, input_shape, output_shapes, anchors):
        self.dataset_bboxes_iterator = dataset_bboxes_iterator
        self.strides = self.compute_strides(input_shape, output_shapes)
        self.output_shapes = output_shapes
        self.anchors = anchors
        self.anchors_per_scale = len(anchors[0])
        self.iterator = iter(self.dataset_bboxes_iterator)

    def compute_strides(self, input_shape, output_shapes):
        # input_shape is (416, 416, 1)
        grid_sizes = np.array([output_shapes[i][1]
                               for i in range(len(output_shapes))])
        return input_shape[0] / grid_sizes

    def __iter__(self):
        return self

    def __next__(self):
        batch_images, batch_bboxes = self.iterator.get_next()
        y_true = self.preprocess_true_bboxes(batch_bboxes)
        return batch_images, y_true

    def preprocess_true_bboxes(self, batch_bboxes):
        """
        The implementation of this function is taken from the
        implementation of YunYang1994 published under the MIT license:
        YunYang1994. Tensorflow-yolov3 [online]. GitHub, 2020 [cit. 2020-8-10].
        Available at: https://github.com/YunYang1994/tensorflow-yolov3
        """
        y_true = [np.zeros((len(batch_bboxes),
                            self.output_shapes[i][1],
                            self.output_shapes[i][2],
                            self.anchors_per_scale, 5)) for i in range(len(self.output_shapes))]

        for image_in_batch, bboxes in enumerate(batch_bboxes):
            # find best anchor for each true bbox
            # (there are 13x13x3 anchors)
            for bbox in bboxes:  # for every true bounding box in the image
                # bbox is [x1, y1, x2, y2]
                bbox_wh = bbox[2:] - bbox[:2]
                # Skip zero sized boxes (if the width or height is 0)
                if np.any(np.isclose(bbox_wh, 0.)):
                    continue
                bbox_center = (bbox[2:] + bbox[:2]) * 0.5
                bbox_xywh = np.concatenate([bbox_center, bbox_wh], axis=-1)
                # transform bbox coordinates into scaled values - for each scale
                # (ie. 13x13 grid box, instead of 416*416 pixels)
                # bbox_xywh_grid_scaled.shape = (2 scales, 4 coords) 
                bbox_xywh_grid_scaled = bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
                # print(bbox_xywh_grid_scaled)
                exist_positive = False
                iou_for_all_scales = []

                # for each scale (13x13 and 26x26)
                for scale_index in range(len(self.output_shapes)):
                    grid_box_xy = np.floor(bbox_xywh_grid_scaled[scale_index, 0:2]).astype(np.int32)
                    # get anchors coordinates for the current 
                    anchors_xywh_scaled = np.zeros((self.anchors_per_scale, 4))
                    # the center of an anchor is the center of a grid box
                    anchors_xywh_scaled[:, 0:2] = grid_box_xy + 0.5
                    # self.anchors defines only widths and heights of anchors
                    # Values of self.anchors should be already scaled.
                    anchors_xywh_scaled[:, 2:4] = self.anchors[scale_index]

                    # compute IOU for true bbox and anchors
                    iou_of_this_scale = bbox_iou(bbox_xywh_grid_scaled[scale_index][np.newaxis, :],
                                                 anchors_xywh_scaled)
                    iou_for_all_scales.append(iou_of_this_scale)
                    iou_mask = iou_of_this_scale > 0.3

                    # update y_true for anchors of grid boxes which satisfy the iou threshold
                    if np.any(iou_mask):
                        x_index, y_index = grid_box_xy

                        y_true[scale_index][image_in_batch, y_index, x_index, iou_mask, :] = 0
                        y_true[scale_index][image_in_batch, y_index, x_index, iou_mask, 0:4] = bbox_xywh
                        y_true[scale_index][image_in_batch, y_index, x_index, iou_mask, 4:5] = 1.0

                        exist_positive = True

                # if no prediction across all scales for the current bbox
                # matched the true bounding box enough
                if not exist_positive:
                    # get the prediction with the highest IOU
                    best_anchor_index = np.argmax(np.array(iou_for_all_scales).reshape(-1), axis=-1)
                    best_detect = int(best_anchor_index / self.anchors_per_scale)
                    best_anchor = int(best_anchor_index % self.anchors_per_scale)
                    x_index, y_index = np.floor(bbox_xywh_grid_scaled[best_detect, 0:2]).astype(np.int32)

                    y_true[best_detect][image_in_batch, y_index, x_index, best_anchor, :] = 0
                    y_true[best_detect][image_in_batch, y_index, x_index, best_anchor, 0:4] = bbox_xywh
                    y_true[best_detect][image_in_batch, y_index, x_index, best_anchor, 4:5] = 1.0
        return y_true


def tf_load_image(image_file_path, dtype, shape):
    """
    Loads an image from file and resizes it with pad to target shape.

    Parameters
    -------
    image_file_path
    dtype
    shape
        An array-like of two values [width, height].


    Returns
    -------
    depth_image
        A 3-D Tensor of shape [height, width, 1].
    """
    depth_image_file_content = tf.io.read_file(image_file_path)

    # loads depth images and converts values to fit in dtype.uint8
    depth_image = tf.io.decode_image(depth_image_file_content, channels=1, dtype=dtype)

    depth_image.set_shape([shape[1], shape[0], 1])
    return depth_image