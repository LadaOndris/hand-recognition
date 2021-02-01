import tensorflow as tf
from tensorflow.keras.layers import Input
from src.core.cfg.cfg_parser import CfgParser


class YoloLayer(tf.keras.layers.Layer):
    def __init__(self, anchors, n_classes, input_layer_shape, name=None):
        super(YoloLayer, self).__init__(name=name)
        self.anchors = anchors
        self.n_anchors = len(anchors)
        self.n_classes = n_classes
        self.input_layer_shape = input_layer_shape

    def call(self, inputs):
        """
        Reshapes inputs to [batch_size, grid_size, grid_size, anchors_per_grid, 6]
        where the axis=-1 contains [x, y, w, h, conf, raw_conf].

        Parameters
        ----------
        inputs : 
            Outputs of previous layer in the model.

        Returns
        -------
        yolo_outputs : Tensor of shape [batch_size, grid_size, grid_size, anchors_per_grid, 5 + n_classes]
            Returns raw predictions [tx, ty, tw, th].
            It is ready for loss calculation, but needs to gor through further postprocessing 
            to convert it to real dimensions.
        """
        # transform to [None, B * grid size * grid size, 5 + C]
        # The B is the number of anchors and C is the number of classes.
        # inputs = tf.reshape(inputs, [-1, self.n_anchors * out_shape[1] * out_shape[2], \
        #                             5 + self.n_classes])

        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        grid_size_y = inputs_shape[1]
        grid_size_x = inputs_shape[2]
        reshaped_inputs = tf.reshape(inputs, [batch_size, grid_size_y, grid_size_x,
                                              self.n_anchors, 5 + self.n_classes])

        # extract information
        box_centers = reshaped_inputs[..., 0:2]
        box_shapes = reshaped_inputs[..., 2:4]
        confidence = reshaped_inputs[..., 4:5]

        # create coordinates for each anchor for each cell
        y = tf.tile(tf.range(grid_size_y, dtype=tf.int32)[:, tf.newaxis], [1, grid_size_y])
        x = tf.tile(tf.range(grid_size_x, dtype=tf.int32)[tf.newaxis, :], [grid_size_x, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, self.n_anchors, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        # for example: 416 x 416 pixel images, 13 x 13 tiles
        # 416 // 13 = 32
        stride = (self.input_layer_shape[1] // grid_size_y,
                  self.input_layer_shape[2] // grid_size_x)

        # get dimensions in pixels instead of grid boxes by multiplying with stride
        pred_xy = (tf.sigmoid(box_centers) + xy_grid) * stride
        pred_wh = (tf.exp(box_shapes) * self.anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(confidence)  # confidence is objectness

        return tf.concat([pred_xywh, pred_conf, confidence], axis=-1)


class Model:

    def __init__(self):
        self.yolo_output_shapes = None
        self.tf_model = None
        self.learning_rate = 1e-3
        self.anchors = []
        self.yolo_out_preds = []
        self.yolo_output_shapes = []

    @classmethod
    def from_cfg(cls, file):
        parser = CfgParser()
        return parser.parse(file)
