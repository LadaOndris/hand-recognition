

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
from datasets.handseg150k.dataset import HandsegDataset, HUGE_INT
from feature_extraction_numba import extract_features, extract_features_and_labels, \
    get_pixel_coords, get_feature_offsets
from joblib import dump, load
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2
from skimage.feature import hog
from skimage import data, exposure
from model import Model
from utils import output_boxes, draw_output_boxes
import tensorflow as tf
from config import Config
from datasets.handseg150k.dataset_bboxes import HandsegDatasetBboxes

# disable CUDA, run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class YoloLoss(tf.keras.losses.Loss):
    
    def __init__(self, model_input_image_size, ignore_thresh=.5, name='yolo_loss'):
        """

        Parameters
        ----------
        model_input_image_size : TYPE
            DESCRIPTION.
        ignore_thresh : TYPE, optional
            DESCRIPTION. The default is .5.
        name : TYPE, optional
            DESCRIPTION. The default is 'yolo_loss'.

        Returns
        -------
        None.

        """
        super(YoloLoss, self).__init__(name=name)
        self.model_input_image_size = model_input_image_size # (416, 416, 1)
        self.ignore_thresh = ignore_thresh
        return
    
    def call(self, y_true, y_pred):
        """
            Computes loss for YOLO network used in object detection.
            
            It doesn't include calculation of class predictions, since a single class is being predicted 
            and confidence is all that is required.
            
            Neither y_true and y_pred should contain class predictions.
        
        Parameters
        ----------
        y_true : 5D array
            True bounding boxes and class labels.
            Shape is (batch_size, grid_size, grid_size, anchors_per_grid_box, 5).
        y_pred : 5D array
            Prediction of bounding boxes and class labels.
            Shape is (batch_size, grid_size, grid_size, anchors_per_grid_box, 6).
            The last dimension also contains raw confidence.
            
        Returns
        -------
        tf.Tensor
            Loss computed for bounding boxes and the confidence whether it contains an object.
            It is used in a YOLO network in object detection.
        """
        
        #tf.print("Y_true", tf.shape(y_true))
        #tf.print("Y_pred", tf.shape(y_pred))
        
        #tf.print("TRUE", y_true[:,6,6,0,...])
        #tf.print("PRED", y_pred[:,6,6,0,...])
        
        # Look out for xywh in different units!!! 
        
        pred_xywh = y_pred[...,0:4]
        pred_conf = y_pred[...,4:5]
        raw_conf = y_pred[...,5:6]
        
        true_xywh = y_true[...,0:4]
        true_conf = y_true[...,4:5]
        
        tf.print("shape", tf.shape(true_xywh))
        #tf.print("pred_xywh, min, max", tf.reduce_min(pred_xywh[...,2:]), tf.reduce_max(pred_xywh[...,2:]))
        tf.print("true_xywh, min, max", tf.reduce_min(true_xywh[...,2:]), tf.reduce_max(true_xywh[...,2:]))
        
        zeros = tf.cast(tf.zeros_like(pred_xywh),dtype=tf.bool)
        ones = tf.cast(tf.ones_like(pred_xywh),dtype=tf.bool)
        loc = tf.where(pred_conf > 0.3, ones, zeros)
        pred_xywh_masked = tf.boolean_mask(pred_xywh, loc)
        tf.print("conf > 0.3: shape, wh_min, wh_max", tf.shape(pred_xywh_masked), 
                 tf.reduce_min(pred_xywh_masked[...,2:]),
                 tf.reduce_max(pred_xywh_masked[...,2:]))

        tf.print("conf mean min max sum true_sum", 
                 tf.reduce_mean(pred_conf), 
                 tf.reduce_min(pred_conf),
                 tf.reduce_max(pred_conf),
                 tf.reduce_sum(pred_conf),
                 tf.reduce_sum(true_conf)) 
        
        xywh_loss = self.iou_loss(true_conf, pred_xywh, true_xywh)
        conf_loss = self.confidence_loss(raw_conf, true_conf, pred_xywh, true_xywh)
        
        # There is no loss for class labels, since there is a single class
        # and confidence score represenets that class
    
        return conf_loss
    
    def xywh_loss(self, true_conf, pred_xywh, true_xywh):
        input_size = tf.cast(self.model_input_image_size[0], tf.float32)
        bbox_loss_scale = 2.0 - true_xywh[..., 2:3] * true_xywh[..., 3:4] / (input_size ** 2)
        
        xy_loss = true_conf * bbox_loss_scale * tf.keras.backend.square(true_xywh[...,:2] - pred_xywh[...,:2])
        wh_loss = true_conf * bbox_loss_scale * 0.5 * tf.keras.backend.square(true_xywh[...,2:] - pred_xywh[...,2:])
        
        return xy_loss + wh_loss
    
        
    def giou_loss(self, true_conf, pred_xywh, true_xywh):
        input_size = tf.cast(self.model_input_image_size[0], tf.float32)
        bbox_loss_scale = 2.0 - true_xywh[..., 2:3] * true_xywh[..., 3:4] / (input_size ** 2)
        
        giou = tf.expand_dims(self.bbox_giou(pred_xywh, true_xywh), axis=-1)
        input_size = tf.cast(self.model_input_image_size, tf.float32)
        #tf.print("giou shape", tf.shape(giou))
        #tf.print("GIOU", tf.reduce_min(giou), tf.reduce_max(giou))
        bbox_loss_scale = 2.0 - 1.0 * true_xywh[:, :, :, :, 2:3] * true_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = true_conf * bbox_loss_scale * (1 - giou)
        return giou_loss
    
        
    def iou_loss(self, true_conf, pred_xywh, true_xywh):
        input_size = tf.cast(self.model_input_image_size[0], tf.float32)
        bbox_loss_scale = 2.0 - true_xywh[..., 2:3] * true_xywh[..., 3:4] / (input_size ** 2)
        
        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], true_xywh[:, :, :, :, np.newaxis, :])
        
        #tf.print("iou shape", tf.shape(iou))
        #tf.print("IOU", tf.reduce_min(iou), tf.reduce_max(iou))
        #tf.print("true_conf.shape, bbox_loss_scale.shape", tf.shape(true_conf), tf.shape(bbox_loss_scale))
        iou_loss = true_conf * bbox_loss_scale * iou
        #tf.print("iou_loss.shape", tf.shape(iou_loss))
        #tf.print("iou_loss", tf.reduce_min(iou_loss), tf.reduce_max(iou_loss))
        return iou_loss
    
    def confidence_loss(self, raw_conf, true_conf, pred_xywh, true_xywh):
        bboxes_mask = true_conf
        #tf.print("bboxes masked", tf.shape(bboxes_mask))
        bboxes_mask = tf.cast(bboxes_mask, dtype=tf.bool)
        bboxes = tf.boolean_mask(true_xywh, bboxes_mask[...,0])
        #tf.print("bboxes masked", tf.shape(bboxes))
        #tf.print("pred_xywh.shape", tf.shape(pred_xywh))
        
        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes)
        # max_iou.shape for example (16, 26, 26, 3, 1)
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1) 
        
        """
        zeros = tf.cast(tf.zeros_like(iou),dtype=tf.bool)
        ones = tf.cast(tf.ones_like(iou),dtype=tf.bool)
        loc = tf.where(iou>0.3, ones, zeros)
        result=tf.boolean_mask(iou, loc)
        tf.print("iou > 0.3", tf.shape(result), result)
        """
        
        """
        From YOLOv3 paper:
        'If the bounding box prior is not the best but does overlap a ground truth object by
        more than some threshold we ignore the prediction.'
        """
        
        ignore_conf = (1.0 - true_conf) * tf.cast(max_iou < self.ignore_thresh, tf.float32)
        conf_loss = \
                true_conf * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_conf, logits=raw_conf) \
              + ignore_conf * tf.nn.sigmoid_cross_entropy_with_logits(labels=true_conf, logits=raw_conf)
        return conf_loss
    
    def bbox_giou(self, boxes1, boxes2):
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        
        giou = iou - (enclose_area - union_area) / enclose_area
        return giou
    
    def bbox_iou(self, boxes1, boxes2):
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou
    

def train():
    
    model = Model.from_cfg("cfg/yolov3-tiny.cfg")
    yolo_out_shapes = model.yolo_output_shapes
    tf_model = model.tf_model
    
    config = Config()
    dataset_bboxes = HandsegDatasetBboxes(batch_size=16)
    dataset_generator = DatasetGenerator(dataset_bboxes.batch_iterator, 
                                         model.input_shape, yolo_out_shapes, model.anchors)
    
    # compile model
    loss = YoloLoss(model.input_shape, ignore_thresh=.5)
    tf_model.compile(optimizer=tf.optimizers.Adam(lr=model.learning_rate), 
                     loss=loss)
    
    tf_model.fit(dataset_generator, epochs=1, verbose=1, steps_per_epoch=1)
    
    model_name = "overfitted_model"
    tf_model.save(model_name)
    
    loaded_model = tf.keras.models.load_model(model_name, custom_objects={'YoloLoss':YoloLoss}, compile=False)
    tf.print(loaded_model)
   
    
    """
    for images, bboxes in dataset.batch_iterator:
        # images.shape is (batch_size, 480, 640, 1)
        # bboxes.shape is (batch_size, 2, 4)
    """
    return

"""
Preprocesses bounding boxes from tf.Dataset and produces y_true.
"""
class DatasetGenerator:
    
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
    
    
    # Mostly taken from https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/dataset.py.
    def preprocess_true_bboxes(self, batch_bboxes):
        y_true = [np.zeros((len(batch_bboxes),
                            self.output_shapes[i][1], 
                            self.output_shapes[i][2], 
                            self.anchors_per_scale, 5)) for i in range(len(self.output_shapes))]
        
        for image_in_batch, bboxes in enumerate(batch_bboxes):
            #print(bboxes)
            # find best anchor for each true bbox
            # (there are 13x13x3 anchors) 
            
            for bbox in bboxes: # for every true bounding box in the image
                # bbox is [x1, y1, x2, y2]
                bbox_center = (bbox[2:] + bbox[:2]) * 0.5
                bbox_wh = bbox[2:] - bbox[:2]
                bbox_xywh = np.concatenate([bbox_center, bbox_wh], axis=-1)
                # transform bbox coordinates into scaled values - for each scale
                # (ie. 13x13 grid box, instead of 416*416 pixels)
                # bbox_xywh_grid_scaled.shape = (2 scales, 4 coords) 
                bbox_xywh_grid_scaled = bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
                #print(bbox_xywh_grid_scaled)
                exist_positive = False
                iou_for_all_scales = []
                
                # for each scale (13x13 and 26x26)
                for scale_index in range(len(self.output_shapes)):
                    grid_box_xy = np.floor(bbox_xywh_grid_scaled[scale_index, 0:2]).astype(np.int32)
                    #print(grid_box_xy)
                    # get anchors coordinates for the current 
                    anchors_xywh_scaled = np.zeros((self.anchors_per_scale, 4))
                    # the center of an anchor is the center of a grid box
                    anchors_xywh_scaled[:, 0:2] = grid_box_xy + 0.5
                    # self.anchors defines only widths and heights of anchors
                    # Values of self.anchors should be already scaled.
                    anchors_xywh_scaled[:, 2:4] = self.anchors[scale_index]
                    
                    # compute IOU for true bbox and anchors
                    iou_of_this_scale = self.bbox_iou(bbox_xywh_grid_scaled[scale_index][np.newaxis,:], anchors_xywh_scaled)
                    iou_for_all_scales.append(iou_of_this_scale) 
                    iou_mask = iou_of_this_scale > 0.3
                    
                    # update y_true for anchors of grid boxes which satisfy the iou threshold
                    if np.any(iou_mask):
                        x_index, y_index = grid_box_xy
                        
                        #iou_mask = [True, False, False]
                        #print(y_true[scale_index][image_in_batch, y_index, x_index, iou_mask, 0:4].shape)
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
            
        print(F"y_true conf sum ({np.sum(y_true[0][...,4:5])}, {np.sum(y_true[1][...,4:5])}):")
        #print(len(y_true), y_true[0].shape)
        return y_true
    
    def bbox_iou(self, boxes1, boxes2):
        """
        boxes1.shape (n, 4)
        boxes2.shape (m, 4)
        
        Returns
            Returns an array of iou for each combination of possible intersection.
        """
        # convert to numpy arrays
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        
        boxes1_area = boxes1[..., 2] * boxes1[..., 3] # width * height
        boxes2_area = boxes2[..., 2] * boxes2[..., 3] # width * height
        
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
    
        return intersection_area / union_area

if __name__ == '__main__':
    train()


