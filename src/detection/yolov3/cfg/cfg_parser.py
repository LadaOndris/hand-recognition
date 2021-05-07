import os
import re
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, Layer, LeakyReLU, MaxPool2D, UpSampling2D, \
    ZeroPadding2D

from src.detection.yolov3.yolo_layer import YoloLayer
from src.detection.yolov3.yolo_model import YoloModel
from src.utils.config import RESIZE_MODE_CROP, RESIZE_MODE_PAD, YOLO_CONFIG_FILE
from src.utils.paths import LOGS_DIR


class YoloLoader:

    def __init__(self):
        self._supported_layers = {
            'net': ['learning_rate', 'batch', 'width', 'height', 'channels'],
            'convolutional': ['filters', 'size', 'stride', 'pad', 'activation', 'batch_normalize'],
            'upsample': ['stride'],
            'maxpool': ['size', 'stride'],
            'route': ['layers'],
            'shortcut': ['from'],
            'yolo': ['classes', 'mask', 'anchors']
        }
        self._custom_layers = {}
        self.throw_error_on_unknown_attr = True
        return

    @classmethod
    def load_from_weights(cls, resize_mode, weights_path=None):
        """
        Loads Tiny YOLOv3 from weights. Initializes a new YoloModel instance
        Parameters
        ----------
        resize_mode     How to treat rectangular images. Allowed options: 'crop', 'pad'.
        weights_path    Uses default weights if not specified.

        Returns
        -------
        yolo_model  An isntance of YoloModel class.
        """
        # Load model based on the preference resize mode
        # because the models were trained with different preprocessing.
        if weights_path is None:
            if resize_mode == RESIZE_MODE_PAD:
                weights_file = "20201016-125612/train_ckpts/ckpt_10"  # mode pad
            elif resize_mode == RESIZE_MODE_CROP:
                weights_file = "20210315-143811/train_ckpts/weights.12.h5"  # mode crop
            else:
                raise ValueError('Invalid resize option!')
            weights_path = LOGS_DIR.joinpath(weights_file)
        loader = YoloLoader()
        model = loader.parse_cfg(YOLO_CONFIG_FILE)
        model.tf_model.load_weights(str(weights_path))
        return model

    def parse_cfg(self, cfg_file_path: str, custom_layers=Dict[str, Tuple[Layer, List[str]]],
                  throw_error_on_unknown_attr=True) -> YoloModel:
        """
        Parses cfg file and returns a model.

        Parameters
        ----------
        cfg_file_path : String
            File containing network architecture in a cfg format.
        custom_layers : Dictionary, optional
            Custom layers in the cfg file. The key is a name of the layer in cfg file
            and the value is a subclass of tf.keras.layers.Layer. The default is {}.


        """
        if not os.path.isfile(cfg_file_path) or not os.access(cfg_file_path, os.R_OK):
            raise OSError(F"File '{cfg_file_path} doesn't exist or is not accessible.'")
        self._custom_layers = custom_layers
        self.throw_error_on_unknown_attr = throw_error_on_unknown_attr

        self._read_file_content(cfg_file_path)
        self._preprocess_file_content()
        self._parse_content_into_blocks()
        self._create_layers_from_blocks()
        self._create_model()
        return self.model

    def _read_file_content(self, cfg_file_path):
        """
        Reads file content and saves all lines into an array.
        """
        with open(cfg_file_path, 'r') as file:
            self._file_content = file.readlines()

    def _preprocess_file_content(self):
        """
        Preprocesses the content in such a way that 
        each line contains a meaningful information and without 
        comments and superfluous whitespaces.
        """

        # Replace all whitespaces from each line.
        for i in range(len(self._file_content)):
            # Replace all whitespaces.
            self._file_content[i] = re.sub(r"\s+", "", self._file_content[i], flags=re.UNICODE)
            # Remove comments.
            self._file_content[i] = self._file_content[i].split('#')[0]

        # Remove all empty lines.
        self._file_content = [line for line in self._file_content if len(line) > 0]

    def _parse_content_into_blocks(self):
        block = {}
        self._blocks = []

        for i, line in enumerate(self._file_content):

            if line[0] == '[':  # indicates new block
                if line[-1] != ']':
                    raise ValueError("Incorrectly enclosed block type." +
                                     "Line cannot start with '[' and not end with ']'.")
                if len(line[1:-1]) == 0:
                    raise ValueError("Missing block type. Found empty brackets '[]'.")

                # Don't append an empty block.
                # But allow empty blocks in cfg file without any properties. 
                # Default values can be used in such cases.
                if i != 0:
                    self._blocks.append(block)  # append the previous block
                    block = {}
                block["type"] = line[1:-1]
            else:  # attribute of a block
                key, value = line.split("=")
                if key not in self._supported_layers[block['type']]:
                    message = F"Unsupported attribute '{key}' in block '{block['type']}' on line {i}."
                    if self.throw_error_on_unknown_attr:
                        raise ValueError(message)
                    else:
                        print("Warning: ", message)
                block[key] = value
        self._blocks.append(block)  # append the last block

    def _create_layers_from_blocks(self):
        if self._blocks[0]['type'] != 'net':
            raise ValueError("The first block has to be the [net] block.")
        self._setup_input_layer(self._blocks[0])

        for i, block in enumerate(self._blocks[1:]):
            block_type = block['type']

            if block_type == 'convolutional':
                self._create_layer_convolutional(block, i)
            elif block_type == 'upsample':
                self._create_layer_upsample(block, i)
            elif block_type == 'maxpool':
                self._create_layer_maxpool(block, i)
            elif block_type == 'route':
                self._create_layer_route(block, i)
            elif block_type == 'shortcut':
                self._create_layer_shortcut(block, i)
            elif block_type == 'yolo':
                self._create_layer_yolo(block, i)
            else:
                raise ValueError(F"Unknown block type {block['type']}.")

            """ Save the outputs of each layer """
            self.outputs.append(self.inputs)

    def _is_block_type_supported(self, block_type: str) -> bool:
        if block_type in self._supported_layers:
            return True
        if block_type in self._custom_layers:
            return True
        if block_type == 'net':
            return True
        return False

    def _setup_input_layer(self, net_block):
        input_shape = (int(net_block['width']), int(net_block['height']), int(net_block['channels']))
        batch_size = int(net_block['batch'])
        learning_rate = float(net_block['learning_rate'])

        self.model = YoloModel()
        self.model.input_shape = input_shape
        self.model.batch_size = batch_size
        self.model.learning_rate = learning_rate
        self.model.anchors = []

        self.inputs = self.input_layer = Input(shape=input_shape, batch_size=batch_size)
        self.outputs = []

    def _create_layer_convolutional(self, block, i: int):
        filters = int(block['filters'])
        kernel_size = int(block['size'])
        stride = int(block['stride'])
        padding = block['pad']
        activation = block['activation']

        try:
            batch_normalize = int(block["batch_normalize"])
            bias = False
        except KeyError:
            batch_normalize = 0
            bias = True

        if padding:
            pad = 'same'
        else:
            pad = 'valid'

        if stride > 1:
            self.inputs = ZeroPadding2D(((1, 0), (1, 0)))(self.inputs)

        self.inputs = Conv2D(filters, kernel_size, strides=(stride, stride),
                             padding=pad, activation=None, use_bias=bias,
                             name=F"conv_{i}")(self.inputs)

        if batch_normalize:
            self.inputs = BatchNormalization(name=F"bnorm_{i}")(self.inputs)

        if activation == 'leaky':
            self.inputs = LeakyReLU(alpha=.1, name=F"leaky_{i}")(self.inputs)
        elif activation != 'linear':
            raise Exception('Unknown activation function')

    def _create_layer_upsample(self, block, i):
        stride = int(block['stride'])
        self.inputs = UpSampling2D(stride, name=F"upsample_{i}")(self.inputs)

    def _create_layer_maxpool(self, block, i):
        size = int(block['size'])
        stride = int(block['stride'])

        self.inputs = MaxPool2D(pool_size=(size, size), strides=(stride, stride), name=F"maxpool_{i}")(self.inputs)

    def _create_layer_route(self, block, i):
        layers = block['layers'].split(',')
        layers = [int(layer) for layer in layers]

        if layers[0] > 0:
            layers[0] = layers[0] - i

        if len(layers) == 1:
            self.inputs = self.outputs[i + layers[0]]
        elif len(layers) == 2:
            if layers[1] > 0:
                layers[1] = layers[1] - i

            # reorganize the previous layer
            target_shape = self.outputs[i + layers[1]].shape
            self.outputs[i + layers[0]] = tf.reshape(self.outputs[i + layers[0]],
                                                     [target_shape[0], target_shape[1], target_shape[2], -1])
            self.inputs = tf.concat([self.outputs[i + layers[0]], self.outputs[i + layers[1]]], axis=-1)
        else:
            raise Exception('Invalid layers')

    def _create_layer_shortcut(self, block, i):
        from_layer = int(block["from"])
        self.inputs = self.outputs[i - 1] + self.outputs[i + from_layer]

    def _create_layer_yolo(self, block, i):
        num_classes = int(block['classes'])
        mask = block['mask'].split(',')
        mask = [int(m) for m in mask]
        anchors = block['anchors'].split(',')
        anchors = [float(a) for a in anchors]
        anchors = [(anchors[i], anchors[i + 1])
                   for i in range(0, len(anchors), 2)]
        anchors = [anchors[m] for m in mask]
        prediction = YoloLayer(anchors, num_classes, self.input_layer.shape, name=F"yolo_{i}")(self.inputs)

        self.model.anchors.append(anchors)
        self.model.yolo_out_preds.append(prediction)
        self.model.yolo_output_shapes.append(prediction.shape)

    def _create_model(self):
        self.model.tf_model = tf.keras.Model(self.input_layer, self.model.yolo_out_preds)


if __name__ == '__main__':
    parser = YoloLoader()
    model = parser.parse_cfg(YOLO_CONFIG_FILE, throw_error_on_unknown_attr=False)
    model.tf_model.summary()
