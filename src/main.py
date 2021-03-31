import matplotlib.pyplot as plt
import tensorflow as tf
from src.detection.yolov3 import utils
from src.core.cfg.cfg_parser import Model
from src.utils.paths import LOGS_DIR, CUSTOM_DATASET_DIR, BIGHAND_DATASET_DIR, MSRAHANDGESTURE_DATASET_DIR
from src.utils.config import TEST_YOLO_CONF_THRESHOLD, YOLO_CONFIG_FILE
from src.utils.live import generate_live_images
from src.pose_estimation.dataset_generator import DatasetGenerator
from src.utils.camera import Camera
import src.utils.plots as plots
from src.pose_estimation.jgr_j2o import JGR_J2O
from src.utils.images import tf_resize_image


class HandPositionEstimator:
    """
    Loads a hand detector and hand pose estimator.
    Then, it uses them to estimate the precision position of hands
    either from files, live images, or from given images.
    """

    def __init__(self, camera: Camera, cube_size, plot_detection=False, plot_estimation=False):
        self.camera = camera
        self.plot_detection = plot_detection
        self.plot_estimation = plot_estimation
        self.resize_mode = 'crop'
        self.detector = self.load_detector(self.resize_mode)
        self.network = JGR_J2O()
        self.estimator = self.load_estimator()
        self.estimation_preprocessor = DatasetGenerator(None, self.detector.input_shape, self.network.input_size,
                                                        self.network.out_size,
                                                        camera=self.camera, augment=False, cube_size=cube_size,
                                                        refine_iters=1)

    def load_detector(self, resize_mode):
        # Load model based on the preference resize mode
        # because the models were trained with different preprocessing.
        if resize_mode == 'pad':
            weights_file = "20201016-125612/train_ckpts/ckpt_10"  # mode pad
        elif resize_mode == 'crop':
            weights_file = "20210315-143811/train_ckpts/weights.12.h5"  # mode crop
        weights_path = LOGS_DIR.joinpath(weights_file)
        model = Model.from_cfg(YOLO_CONFIG_FILE)
        model.tf_model.load_weights(str(weights_path))
        return model

    def load_estimator(self):
        # Load HPE model and weights

        # weights_path = LOGS_DIR.joinpath('20210329-032745/train_ckpts/weights.04.h5')  # bighand
        # weights_path = LOGS_DIR.joinpath('20210316-035251/train_ckpts/weights.18.h5')
        # weights_path = LOGS_DIR.joinpath("20210323-160416/train_ckpts/weights.10.h5")
        weights_path = LOGS_DIR.joinpath("20210330-024055/train_ckpts/weights.52.h5")

        model = self.network.graph()
        model.load_weights(str(weights_path))
        return model

    def inference_from_file(self, file_path: str):
        """
        Detect a hand from a single image.
        """
        image = self.read_image(file_path)
        image = self.resize_image_and_depth(image)
        return self.inference_from_image(image)

    def inference_live(self):
        live_image_generator = generate_live_images()

        for depth_image in live_image_generator:
            depth_image = self.resize_image_and_depth(depth_image)
            joints = self.inference_from_image(depth_image)

    def inference_from_image(self, image):
        """
        Performs hand detection and pose estimation on the given image.

        Parameters
        ----------
        image
            Image pixels should be in milimeters as they are preprocessed
            accordingly for the detector.

        Returns
        -------

        """
        batch_images = tf.expand_dims(image, axis=0)
        boxes = self.detect(batch_images)

        # If detection failed
        if tf.experimental.numpy.allclose(boxes, 0):
            return None
        joints_uvz = self.estimate(batch_images, boxes)
        return joints_uvz

    def detect_live(self):
        """
        Reads live images from RealSense depth camera, and
        detects hands for each frame.
        """
        # create live image generator
        live_image_generator = generate_live_images()

        for depth_image in live_image_generator:
            depth_image = self.resize_image_and_depth(depth_image)
            batch_images = tf.expand_dims(depth_image, axis=0)
            self.detect(batch_images)

    def detect(self, images):
        """

        Parameters
        ----------
        images

        Returns
        -------
        boxes : shape [batch_size, 4]
            Returns all zeros if non-max suppression did not find any valid boxes.
        """
        detection_batch_images = preprocess_image_for_detection(images)
        yolo_outputs = self.detector.tf_model.predict(detection_batch_images)

        boxes, scores, nums = utils.boxes_from_yolo_outputs(yolo_outputs, self.detector.batch_size,
                                                            self.detector.input_shape,
                                                            TEST_YOLO_CONF_THRESHOLD, iou_thresh=.7, max_boxes=1)
        if self.plot_detection:
            utils.draw_predictions(images[0], boxes[0], nums[0], fig_location=None)
        return boxes

    def estimate(self, images, boxes):
        boxes = tf.cast(boxes, dtype=tf.int32)
        normalized_images = self.estimation_preprocessor.preprocess(images, boxes[:, 0, :])
        y_pred = self.estimator.predict(normalized_images)
        uvz_pred = self.estimation_preprocessor.postprocess(y_pred)

        # Plot the inferenced pose
        if self.plot_estimation:
            joints2d = uvz_pred[..., :2] - self.estimation_preprocessor.bboxes[..., tf.newaxis, :2]
            plots.plot_joints_2d(self.estimation_preprocessor.cropped_imgs[0].to_tensor(), joints2d[0])
        return uvz_pred

    def read_image(self, file_path: str):
        """
        Reads image from file.

        Returns
        -------
        Depth image
        """
        # load image
        depth_image = utils.tf_load_image(file_path, dtype=tf.uint16, shape=self.camera.image_size)
        return depth_image

    def resize_image_and_depth(self, image):
        """
        Resizes to size matching the detector's input shape.
        The unit of image pixels are set as milimeters.
        """
        image = tf_resize_image(image, shape=self.detector.input_shape[:2],
                                resize_mode=self.resize_mode)
        image = set_depth_unit(image, 0.001, self.camera.depth_unit)
        return image


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


if __name__ == '__main__':
    dataset = 'custom'
    estimator = HandPositionEstimator(Camera(dataset), cube_size=200, plot_detection=False, plot_estimation=True)

    if dataset == 'bighand':
        for i in range(3):
            for j in range(3):
                estimator.inference_from_file(str(BIGHAND_DATASET_DIR.joinpath(
                    F"Subject_2/1 75/image_D000{i}{j}000.png")))
    else:
        estimator.inference_from_file(str(CUSTOM_DATASET_DIR.joinpath('20210326-230536/85.png')))
        estimator.inference_from_file(str(CUSTOM_DATASET_DIR.joinpath('20210326-232147/42.png')))
        # estimator.inference_from_file(str(CUSTOM_DATASET_DIR.joinpath('20210326-232751/420.png')))

        # for j in range(9):
        #    estimator.inference_from_file(str(CUSTOM_DATASET_DIR.joinpath(F"20210326-232751/70{j}.png")))

    # estimator.detect_live()

    pass
