
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

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
    tf.print(tf.reduce_mean(bboxes))
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
    
    tf.print(tf.reduce_mean(boxes))
    return boxes, scores, num_valid_boxes


def output_boxes(inputs, model_size, max_output_size, max_output_size_per_class, 
                 iou_threshold, confidence_threshold):
    
    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)
    
    top_left_x = center_x - width * 0.5
    tf.print("min max", tf.reduce_min(top_left_x), tf.reduce_min(center_x))
    tf.print("Well", center_x[0], width[0], top_left_x[0])
    top_left_y = center_y - height * 0.5
    bottom_right_x = center_x + width * 0.5
    bottom_right_y = center_y + height * 0.5
    
    inputs = tf.concat([top_left_x, 
                        top_left_y, 
                        bottom_right_x,
                        bottom_right_y, 
                        confidence, classes], axis=-1)
    
    print("Non max suppression")
    boxes_dicts = non_max_suppression(inputs, model_size, max_output_size, \
        max_output_size_per_class, iou_threshold, confidence_threshold)
    
    return boxes_dicts

def draw_output_boxes(image, boxes, nums):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    for i in range(nums):
        x, y = boxes[i,0:2]# * [image.shape[1],image.shape[0]]
        w, h = boxes[i,2:4] - boxes[i,0:2]# * [image.shape[1],image.shape[0]]
        print(x, y, w, h)
        rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    
    plt.show()
    return

conf_thresh = .4

def draw_detected_objects(images, predictions_for_the_image, model_size):
    
    boxes, scores, nums = output_boxes(predictions_for_the_image, model_size, 5, 2, .5, conf_thresh)
    
    #tf.print("In draw detected", tf.reduce_mean(boxes))
    for i in range(len(boxes)):
        print("Drawing boxes with scores:", scores[i][:nums[i]])
        draw_output_boxes(images[i], boxes[i], nums[i])
        
        
def draw_grid_detection(images, yolo_outputs, model_size):
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
    
    
    
