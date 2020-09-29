
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
from utils import output_boxes, draw_output_boxes, draw_detected_objects, draw_grid_detection
import tensorflow as tf
from config import Config
from datasets.handseg150k.dataset_bboxes import HandsegDatasetBboxes
from datasets.simple_boxes.dataset_bboxes import SimpleBoxesDataset
from dataset_generator import DatasetGenerator

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
    
        return xywh_loss + conf_loss
    
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
    
batch_size = 1

def train():
    
    model = Model.from_cfg("cfg/yolov3-tiny.cfg")
    yolo_out_shapes = model.yolo_output_shapes
    tf_model = model.tf_model
    
    #config = Config()
    dataset_bboxes = SimpleBoxesDataset(batch_size=batch_size)
    dataset_generator = DatasetGenerator(dataset_bboxes.batch_iterator, 
                                         model.input_shape, yolo_out_shapes, model.anchors)
    
    # compile model
    loss = YoloLoss(model.input_shape, ignore_thresh=.5)
    tf_model.compile(optimizer=tf.optimizers.Adam(lr=model.learning_rate), 
                     loss=loss)
    
    tf_model.fit(dataset_generator, epochs=1, verbose=1, steps_per_epoch=312)
    
    model_name = "simple_boxes.h5"
    tf_model.save_weights(model_name)
    
    
    """
    for images, bboxes in dataset_bboxes.batch_iterator:
        # images.shape is (batch_size, 416, 416, 1)
        # bboxes.shape is (batch_size, 2, 4)
    """
    return

def predict():
    
    model = Model.from_cfg("cfg/yolov3-tiny.cfg")
    loaded_model = model.tf_model
    loaded_model.load_weights("simple_boxes.h5")
    #loaded_model = tf.keras.models.load_model("overfitted_model_conf_only", custom_objects={'YoloLoss':YoloLoss}, compile=False)
    
    dataset_bboxes = SimpleBoxesDataset(batch_size=batch_size)
    #yolo_outputs = loaded_model.predict(dataset_bboxes, batch_size=16, steps=1, verbose=1)
    
    i = 0
    for batch_images, batch_bboxes in dataset_bboxes.batch_iterator:
        print(batch_images.shape)
        yolo_outputs = loaded_model.predict(batch_images)
        scale1_outputs = tf.reshape(yolo_outputs[0], [batch_size, -1, 6])
        scale2_outputs = tf.reshape(yolo_outputs[1], [batch_size, -1, 6])
        outputs = tf.concat([scale1_outputs, scale2_outputs], axis=1) # outputs for the whole batch
        
        #single_image_preds = outputs[0]
        #image = np.squeeze(batch_images[0])
        
        draw_grid_detection(batch_images, yolo_outputs, [416, 416, 1])
        draw_detected_objects(batch_images, outputs, [416, 416, 1])
        
        i += 1
        if i == 10:
            break


if __name__ == '__main__':
    predict()


