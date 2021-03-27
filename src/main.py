"""
1. detection
2. pose estimation
3. gesture fulfillment
"""

# from detection.rdf import load_model
import detection
import pose_estimation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import os
from PIL import Image
from src.detection.yolov3 import utils
from src.core.cfg.cfg_parser import Model
from src.utils.paths import LOGS_DIR, SRC_DIR, CUSTOM_DATASET_DIR
from src.utils.config import TEST_YOLO_CONF_THRESHOLD, YOLO_CONFIG_FILE
from src.utils.live import generate_live_images
from src.utils.paths import OTHER_DIR
from src.pose_estimation.preprocessing import ComPreprocessor
from src.pose_estimation.dataset_generator import DatasetGenerator
from src.utils.camera import Camera
import src.utils.plots as plots
from src.pose_estimation.jgr_j2o import JGR_J2O


def load_detector():
    model = Model.from_cfg(YOLO_CONFIG_FILE)

    # Also change resize mode down below
    # because the model was trained with different preprocessing mode
    # weights_file = "20201016-125612/train_ckpts/ckpt_10"  # mode pad
    weights_file = "20210315-143811/train_ckpts/weights.12.h5"  # mode crop
    weights_path = LOGS_DIR.joinpath(weights_file)
    model.tf_model.load_weights(str(weights_path))
    return model


def load_estimator(network):
    # Load HPE model and weights
    weights_path = LOGS_DIR.joinpath('20210316-035251/train_ckpts/weights.18.h5')
    # weights_path = LOGS_DIR.joinpath("20210323-160416/train_ckpts/weights.10.h5")
    model = network.graph()
    model.load_weights(str(weights_path))
    return model


def detect_live():
    """
    Reads live images from RealSense depth camera, and
    detects hands for each frame.
    """
    # create live image generator
    live_image_generator = generate_live_images()
    # load detection model
    model = load_detector()

    while True:
        # load image
        depth_image = next(live_image_generator)
        depth_image = utils.tf_resize_image(depth_image)

        # create a batch with a single image
        batch_images = tf.expand_dims(depth_image, axis=0)

        # predict
        yolo_outputs = model.tf_model.predict(batch_images)

        # show result
        # utils.draw_grid_detection(batch_images, yolo_outputs, [416, 416, 1], TEST_YOLO_CONF_THRESHOLD)
        utils.draw_detected_objects(batch_images, yolo_outputs, [416, 416, 1], TEST_YOLO_CONF_THRESHOLD)


def read_image_to_batch(file_path: str, camera: Camera):
    """

    Parameters
    ----------
    file_path
    camera

    Returns
    -------
        Reads image from file, and resizes to size [416, 416, 1].
        Expands dims along the first axis to create a batch.
    """
    preprocess_image_size = [416, 416, 1]
    # load image
    depth_image = utils.tf_load_image(file_path, dtype=tf.uint16, shape=camera.image_size)
    depth_image = utils.tf_resize_image(depth_image, shape=preprocess_image_size[:2], resize_mode='crop')
    depth_image = set_depth_unit(depth_image, 0.001, camera.depth_unit)
    # create a batch with a single image
    batch_images = tf.expand_dims(depth_image, axis=0)
    return batch_images


def set_depth_unit(images, target_depth_unit, previous_depth_unit):
    """
    Converts image pixel values to the specified unit.
    """
    dtype = images.dtype
    images = tf.cast(images, dtype=tf.float32)
    images *= previous_depth_unit / target_depth_unit
    images = tf.cast(images, dtype=dtype)
    return images


def preprocess_image_for_detection(images):
    """
    Multiplies pixel values by 8 to match the units expected by the detector.
    Converts image dtype to tf.uint8.

    Parameters
    ----------
    images
        Image pixel values should be in milimeters.
    camera
    """
    dtype = images.dtype
    images = tf.cast(images, dtype=tf.float32)
    images *= 8.00085466544
    images = tf.cast(images, dtype=dtype)
    images = tf.image.convert_image_dtype(images, dtype=tf.uint8)
    return images


def detect_from_file(file_path: str, camera: Camera):
    """
    Detect a hand from a single image.
    """
    # ------ Read image ------
    batch_images = read_image_to_batch(file_path, camera)

    # ------ Pass to detection stage ------
    detection_batch_images = preprocess_image_for_detection(batch_images)
    detector = load_detector()
    yolo_outputs = detector.tf_model.predict(detection_batch_images)

    # utils.draw_grid_detection(batch_images, yolo_outputs, preprocess_image_size, TEST_YOLO_CONF_THRESHOLD)
    # utils.draw_detected_objects(batch_images, yolo_outputs, preprocess_image_size, TEST_YOLO_CONF_THRESHOLD,
    #                            max_boxes=1)

    boxes, scores, nums = utils.boxes_from_yolo_outputs(yolo_outputs, detector.batch_size, detector.input_shape,
                                                        TEST_YOLO_CONF_THRESHOLD, iou_thresh=.7, max_boxes=1)
    utils.draw_predictions(batch_images[0], boxes[0], nums[0], fig_location=None)

    # ------ Pass to hand pose estimation stage ------
    # Prepare HPE model
    network = JGR_J2O()
    estimator = load_estimator(network)

    # Preprocess the image
    camera = Camera('custom')
    dataset_generator = DatasetGenerator(None, detector.input_shape, network.input_size, network.out_size,
                                         camera=camera, augment=False, cube_size=180)
    boxes = tf.cast(boxes, dtype=tf.int32)
    normalized_images = dataset_generator.preprocess(batch_images, boxes[:, 0, :])

    # Inference
    y_pred = estimator.predict(normalized_images)

    # Postprocess inferenced pose
    uvz_pred = dataset_generator.postprocess(y_pred)
    joints2d = uvz_pred[..., :2] - dataset_generator.bboxes[..., tf.newaxis, :2]

    # Plot the inferenced pose
    plots.plot_joints_2d(normalized_images[0], joints2d[0])

    pass


if __name__ == '__main__':
    cam = Camera('sr305')
    # detect_from_file(str(CUSTOM_DATASET_DIR.joinpath('20210326-153934/35.png')), Camera('d415'))
    detect_from_file(str(CUSTOM_DATASET_DIR.joinpath('20210326-230536/47.png')), Camera('sr305'))
    # detect_from_file(str(OTHER_DIR.joinpath('me.png')))
    # detect_live()
    pass
