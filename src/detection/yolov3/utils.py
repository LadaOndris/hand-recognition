
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


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

def tensorflow_bbox_giou(boxes1, boxes2):
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

# inspired by https://machinelearningspace.com/yolov3-tensorflow-2-part-4/
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
    #bboxes = bboxes / model_size[0]
    #tf.print(tf.reduce_mean(bboxes))
    bboxes = tf.expand_dims(bboxes, axis=2) # the third dimension is 1 according to documentation
    
    boxes, scores, classes, num_valid_boxes = tf.image.combined_non_max_suppression(
        boxes=bboxes,
        scores=confs,
        max_output_size_per_class=max_output_size_per_class,
        max_total_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=confidence_threshold,
        clip_boxes=False
    )
    
    #tf.print(tf.reduce_mean(boxes))
    return boxes, scores, num_valid_boxes


def output_boxes(inputs, model_size, max_output_size, max_output_size_per_class, 
                 iou_threshold, confidence_threshold):
    
    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)
    
    top_left_x = center_x - width * 0.5
    #tf.print("min max", tf.reduce_min(top_left_x), tf.reduce_min(center_x))
    #tf.print("Well", center_x[0], width[0], top_left_x[0])
    top_left_y = center_y - height * 0.5
    bottom_right_x = center_x + width * 0.5
    bottom_right_y = center_y + height * 0.5
    
    inputs = tf.concat([top_left_x, 
                        top_left_y, 
                        bottom_right_x,
                        bottom_right_y, 
                        confidence, classes], axis=-1)
    
    #print("Non max suppression")
    boxes_dicts = non_max_suppression(inputs, model_size, max_output_size, \
        max_output_size_per_class, iou_threshold, confidence_threshold)
    
    return boxes_dicts

def draw_output_boxes(image, boxes, nums):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    for i in range(nums):
        x, y = boxes[i,0:2]# * [image.shape[1],image.shape[0]]
        w, h = boxes[i,2:4] - boxes[i,0:2]# * [image.shape[1],image.shape[0]]
        
        rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    
    plt.show()
    return



def draw_detected_objects(images, yolo_outputs, model_size, conf_thresh):
    batch_size = tf.shape(images)[0]
    
    scale1_outputs = tf.reshape(yolo_outputs[0], [batch_size, -1, 6])
    scale2_outputs = tf.reshape(yolo_outputs[1], [batch_size, -1, 6])
    predictions_for_the_image = tf.concat([scale1_outputs, scale2_outputs], axis=1) # outputs for the whole batch
    
    boxes, scores, nums = output_boxes(predictions_for_the_image, model_size, 5, 1, .3, conf_thresh)
    
    #tf.print("In draw detected", tf.reduce_mean(boxes))
    for i in range(len(boxes)):
        print("Drawing boxes with scores:", scores[i][:nums[i]])
        draw_output_boxes(images[i], boxes[i], nums[i])
        
        
def draw_grid_detection(images, yolo_outputs, model_size, conf_thresh):
    """
    Draws images and highlights grid boxes where the model is quite certain 
    that it overlaps an object (the grid box is reponsible for that object prediction).

    Parameters
    ----------
    images : TYPE
        DESCRIPTION.
    yolo_outputs : TYPE
        Boxes defined as (x, y, w, h) where x, y are box centers coordinates
        and w, h their width and height.
    model_size : TYPE
        Image size.

    Returns
    -------
    None.

    """
    for i in range(len(images)):
        fig, ax = plt.subplots(1)
        ax.imshow(images[i])
        
        
        for scale in range(len(yolo_outputs)):
            outputs = yolo_outputs[scale]
            outputs_shape = tf.shape(outputs)
            grid_size = outputs_shape[1:3]
            stride = model_size[0] / grid_size[0]
            
            #tf.print("min max pred", tf.reduce_min(outputs[i,...,4]), tf.reduce_max(outputs[i,...,4]))
            
            #pred_xywh, pred_conf, pred_conf_raw = tf.split(outputs, [4,1,1,], axis=-1)
            for y in range(grid_size[0]):
                for x in range(grid_size[1]):
                    mask = outputs[i, y, x, :, 4:5] > conf_thresh
                    if np.any(mask):
                        rect = patches.Rectangle((x*stride,y*stride),stride,stride,linewidth=1,edgecolor='r',facecolor='none')
                        ax.add_patch(rect)
        
        plt.show()
    

def tf_load_preprocessed_image(image_file_path, shape = [416, 416]):
    """
    Loads an image from file and resizes it with pad to target shape.

    Returns
    -------
    depth_image
        A 3-D Tensor of shape [shape[0], shape[1], 1].

    """
    depth_image_file_content = tf.io.read_file(image_file_path)
    
    # loads depth images and converts values to fit in dtype.uint8
    depth_image = tf.io.decode_image(depth_image_file_content, channels=1)
    depth_image.set_shape([480, 640, 1]) 
    #depth_image /= 255 # normalize to range [0, 1]
    
    return tf_preprocess_image(depth_image)

def tf_preprocess_image(depth_image, shape = [416, 416]):
    # convert the values to range 0-255 as tf.io.read_file does
    depth_image = tf.image.convert_image_dtype(depth_image, dtype=tf.uint8)
    # resize image
    depth_image = tf.image.resize_with_pad(depth_image, shape[0], shape[1])
    
    tf.print("MIN MAX: ", tf.reduce_min(depth_image), tf.reduce_max(depth_image), tf.reduce_mean(depth_image))
    return depth_image
    
    
