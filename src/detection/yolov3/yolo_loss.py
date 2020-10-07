
import tensorflow as tf
import numpy as np

from src.detection.yolov3 import utils

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
            The last dimension also contains raw confidence.(tf.exp(box_shapes) * self.anchors) * stride
            
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
        
        #tf.print("shape", tf.shape(true_xywh))
        #tf.print("pred_xywh, min, max", tf.reduce_min(pred_xywh[...,2:]), tf.reduce_max(pred_xywh[...,2:]))
        #tf.print("true_xywh, min, max", tf.reduce_min(true_xywh[...,2:]), tf.reduce_max(true_xywh[...,2:]))
        
        zeros = tf.cast(tf.zeros_like(pred_xywh),dtype=tf.bool)
        ones = tf.cast(tf.ones_like(pred_xywh),dtype=tf.bool)
        loc = tf.where(pred_conf > 0.3, ones, zeros)
        pred_xywh_masked = tf.boolean_mask(pred_xywh, loc)
        #tf.print("conf > 0.3: shape, wh_min, wh_max", tf.shape(pred_xywh_masked), 
        #         tf.reduce_min(pred_xywh_masked[...,2:]),
        #         tf.reduce_max(pred_xywh_masked[...,2:]))

        #tf.print("conf max sum true_sum", 
        #         tf.reduce_max(pred_conf),
        #         tf.reduce_sum(pred_conf),
        #         tf.reduce_sum(true_conf)) 
        
        xywh_loss = self.iou_loss(true_conf, pred_xywh, true_xywh)
        conf_loss = self.confidence_loss(raw_conf, true_conf, pred_xywh, true_xywh)
        
        # There is no loss for class labels, since there is a single class
        # and confidence score represenets that class
    
        return xywh_loss + conf_loss
    
    def xywh_loss(self, true_conf, pred_xywh, true_xywh):
        input_size = tf.cast(self.model_input_image_size[0], tf.float32)
        bbox_loss_scale = 2.0 - true_xywh[..., 2:3] * true_xywh[..., 3:4] / (input_size ** 2)
        
        # wh = (tf.exp(box_shapes) * self.anchors) * stride
        #strides = self.model_input_image_size[0] / true_xywh.shape[1]
        #pred_wh_inversed = tf.math.log(pred_xywh[2:] / (strides * self.anchors))
        #true_wh_inversed = tf.math.log(true_xywh[2:] / (strides * self.anchors))
        
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
        
        # Big boxes will generaly have greater IOU than small boxes.
        # We need to compensate for the different box sizes, 
        # so that the model is trained equally for small boxes and big boxes.
        bbox_loss_scale = 2.0 - true_xywh[..., 2:3] * true_xywh[..., 3:4] / (input_size ** 2)
        iou = utils.tensorflow_bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], true_xywh[:, :, :, :, np.newaxis, :])
        
        # If IOU = 0, then the boxes don't overlap - the worst result.
        # If IOU = 1, then the boxes overlap exactly - the best result.
        iou_loss = true_conf * bbox_loss_scale * (1 - iou) 
        
        #tf.print("iou shape", tf.shape(iou))
        #tf.print("IOU", tf.reduce_min(iou), tf.reduce_max(iou))
        #tf.print("true_conf.shape, bbox_loss_scale.shape", tf.shape(true_conf), tf.shape(bbox_loss_scale))
        
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
        
        iou = utils.tensorflow_bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes)
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