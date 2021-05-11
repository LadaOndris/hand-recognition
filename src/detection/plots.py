import numpy as np
import tensorflow as tf
from matplotlib import patches as patches, pyplot as plt

from src.detection.yolov3.utils import blue_color, boxes_color, boxes_from_yolo_outputs, depth_image_cmap, \
    prediction_box_color
from src.utils.plots import plotlive, save_show_fig


def plot_predictions(image, boxes, nums, fig_location):
    fig, ax = image_plot()
    _plot_predictions(fig, ax, image, boxes, nums)
    save_show_fig(fig, fig_location, True)


@plotlive
def plot_predictions_live(fig, ax, image, boxes, nums):
    _plot_predictions(fig, ax, image, boxes, nums)


def _plot_predictions(fig, ax, image, boxes, nums):
    ax.imshow(image, cmap=depth_image_cmap)

    for i in range(nums):
        x, y = boxes[i, 0:2].numpy()
        w, h = (boxes[i, 2:4] - boxes[i, 0:2]).numpy()
        plot_prediction_box(ax, x, y, w, h)

    plot_adjust(fig, ax)


def plot_adjust(fig, ax):
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)


def plot_predictions_with_cells(image, boxes, nums, stride, fig_location=None):
    fig, ax = image_plot()
    ax.imshow(image, cmap=depth_image_cmap)

    for i in range(nums):
        x, y = boxes[i, 0:2].numpy()
        w, h = (boxes[i, 2:4] - boxes[i, 0:2]).numpy()

        centroid = (x + w / 2, y + h / 2)
        plot_prediction_box(ax, x, y, w, h)
        plot_responsible_cell(ax, centroid, stride)
        plot_centroid(ax, centroid)

    plot_adjust(fig, ax)
    save_show_fig(fig, fig_location, True)


def plot_prediction_box(ax, x, y, w, h):
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=prediction_box_color, facecolor='none')
    ax.add_patch(rect)


def plot_centroid(ax, centroid):
    centroid = patches.Circle((centroid[0], centroid[1]), radius=4, facecolor=prediction_box_color)
    ax.add_patch(centroid)


def plot_responsible_cell(ax, centroid, stride):
    rect = patches.Rectangle((centroid[0] // stride * stride, centroid[1] // stride * stride), stride, stride,
                             linewidth=2, edgecolor=blue_color, facecolor='none')
    ax.add_patch(rect)


def plot_detected_objects(images, yolo_outputs, model_size, conf_thresh, max_boxes=2, iou_thresh=.5,
                          draw_cells=False, fig_location=None):
    """
    Computes NMS and plots the detected objects.
    """
    batch_size = tf.shape(images)[0]
    boxes, scores, nums = boxes_from_yolo_outputs(yolo_outputs, batch_size, model_size, conf_thresh,
                                                  max_boxes=max_boxes, iou_thresh=iou_thresh)

    # Use the first scale (just for the plot).
    outputs = yolo_outputs[0]
    outputs_shape = tf.shape(outputs)
    grid_size = outputs_shape[1:3]
    stride = (model_size[0] / grid_size[0]).numpy()

    for i in range(len(boxes)):
        print("Drawing boxes with scores:", scores[i][:nums[i]])
        if fig_location:
            fig_location = fig_location.format(i)

        if draw_cells:
            plot_predictions_with_cells(images[i], boxes[i], nums[i], stride, fig_location)
        else:
            plot_predictions(images[i], boxes[i], nums[i], fig_location)


def plot_grid_detection(images, yolo_outputs, model_size, conf_thresh, fig_location=None):
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
        fig, ax = image_plot()
        ax.imshow(images[i], cmap=depth_image_cmap)

        for scale in range(len(yolo_outputs)):
            outputs = yolo_outputs[scale]
            outputs_shape = tf.shape(outputs)
            grid_size = outputs_shape[1:3]
            stride = model_size[0] / grid_size[0]

            # tf.print("min max pred", tf.reduce_min(outputs[i,...,4]), tf.reduce_max(outputs[i,...,4]))

            # pred_xywh, pred_conf, pred_conf_raw = tf.split(outputs, [4,1,1,], axis=-1)
            for y in range(grid_size[0]):
                for x in range(grid_size[1]):
                    mask = outputs[i, y, x, :, 4:5] > conf_thresh
                    if np.any(mask):
                        rect = patches.Rectangle(((x * stride).numpy(), (y * stride).numpy()),
                                                 stride.numpy(), stride.numpy(), linewidth=1, edgecolor=boxes_color,
                                                 facecolor='none')
                        ax.add_patch(rect)
        plot_adjust(fig, ax)
        save_show_fig(fig, fig_location, True)


def plot_grid(images, yolo_outputs, model_size, fig_location=None):
    for i in range(len(images)):
        for scale in range(len(yolo_outputs)):
            fig, ax = image_plot()
            ax.imshow(images[i], cmap=depth_image_cmap)

            outputs = yolo_outputs[scale]
            outputs_shape = tf.shape(outputs)
            grid_size = outputs_shape[1:3]
            stride = model_size[0] / grid_size[0]

            for y in range(grid_size[0]):
                for x in range(grid_size[1]):
                    rect = patches.Rectangle(((x * stride).numpy(), (y * stride).numpy()),
                                             stride.numpy(), stride.numpy(), linewidth=1, edgecolor=boxes_color,
                                             facecolor='none')
                    ax.add_patch(rect)

            plot_adjust(fig, ax)
            save_show_fig(fig, fig_location, True)
            break


def image_plot():
    return plt.subplots(1, figsize=(4, 4))
