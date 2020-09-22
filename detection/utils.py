
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def non_max_suppression(inputs, model_size, max_output_size, 
                        max_output_size_per_class, iou_threshold, confidence_threshold):
    
    bbox, confs, class_probs = tf.split(inputs, [4, 1, -1], axis=-1)
    bbox = bbox / model_size[0]
    
    scores = confs #* class_probs
    print("tf.image.combined_non_max_suppression")
    boxes, scores, classes, num_valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=max_output_size_per_class,
        max_total_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=confidence_threshold
    )
    print("returning from non_max_suppression")
    return boxes, scores, classes, num_valid_detections


def output_boxes(inputs, model_size, max_output_size, max_output_size_per_class, 
                 iou_threshold, confidence_threshold):
    
    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)
    
    top_left_x = center_x - width / 2.0
    top_left_y = center_y - height / 2.0
    bottom_right_x = center_x + width / 2.0
    bottom_right_y = center_y + height / 2.0
    
    inputs = tf.concat([top_left_x, 
                        top_left_y, 
                        bottom_right_x,
                        bottom_right_y, 
                        confidence, classes], axis=-1)
    
    print("Non max suppression")
    boxes_dicts = non_max_suppression(inputs, model_size, max_output_size, \
        max_output_size_per_class, iou_threshold, confidence_threshold)
    
    return boxes_dicts

def draw_output_boxes(image, boxes, scores, classes, nums):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    for i in range(nums):
        x, y = boxes[i,0:2] * [image.shape[1],image.shape[0]]
        w, h = boxes[i,2:4] * [image.shape[1],image.shape[0]]
        print(x, y, w, h)
        rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    
    plt.show()
    return