
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, \
    LeakyReLU, ZeroPadding2D, UpSampling2D, MaxPool2D


class YoloLayer(tf.keras.layers.Layer):
    def __init__(self, anchors, n_classes, input_layer_shape, name=None):
        super(YoloLayer, self).__init__(name=name)
        self.anchors = anchors
        self.n_anchors = len(anchors)
        self.n_classes = n_classes
        self.input_layer_shape = input_layer_shape
            
        
    def call(self, inputs):
        # transform to [None, B * grid size * grid size, 5 + C]
        # The B is the number of anchors and C is the number of classes.
        #inputs = tf.reshape(inputs, [-1, self.n_anchors * out_shape[1] * out_shape[2], \
        #                             5 + self.n_classes])
        
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        output_size = inputs_shape[1]
        inputs = tf.reshape(inputs, [batch_size, output_size, output_size, \
                                     self.n_anchors, 5 + self.n_classes])
        
        # extract information
        box_centers = inputs[..., 0:2]
        box_shapes = inputs[..., 2:4]
        confidence = inputs[..., 4:5]
        #classes = inputs[..., 5:self.n_classes + 5]
        
        
        # create coordinates for each anchor for each cell
        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
        
        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, self.n_anchors, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)
        
        
        # for example: 416 x 416 pixel images, 13 x 13 tiles
        # 416 // 13 = 32
        stride = (self.input_layer_shape[1] // output_size, \
                   self.input_layer_shape[2] // output_size)
            
        
        pred_xy = (tf.sigmoid(box_centers) + xy_grid) * stride
        pred_wh = tf.exp(box_shapes) * tf.cast(self.anchors, dtype=tf.float32)
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        
        pred_conf = tf.sigmoid(confidence) # confidence is objectness
        #pred_class = tf.sigmoid(classes) # classes are class predictions
        
        prediction = tf.concat([pred_xywh, pred_conf], axis=-1)
        return prediction


class Model:
    
    def __init__(self):
        self.yolo_output_shapes = None
        self.tf_model = None
        self.learning_rate = 1e-3
        return
    
    @property
    def yolo_output_shapes(self):
        return self._yolo_output_shapes
    
    @yolo_output_shapes.setter
    def yolo_output_shapes(self, value):
        self._yolo_output_shapes = value
        
    @property
    def tf_model(self):
        return self._tf_model
    
    @tf_model.setter
    def tf_model(self, value):
        self._tf_model = value
    
    @classmethod
    def from_cfg(cls, cfg_file_path):
        blocks = Model.parse_cfg(cfg_file_path)
        net_info_block = blocks[0]
        model = Model.create_model_from_blocks(blocks)
        model.learning_rate = float(net_info_block['learning_rate'])
        return model
    
    @classmethod
    def parse_cfg(cls, cfgfile):
        lines = Model.get_preprocessed_cfg(cfgfile)
    
        block = {}
        blocks = []
        
        for line in lines:
            if line[0] == "[":               # This marks the start of a new block
                if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                    blocks.append(block)     # add it the blocks list
                    block = {}               # re-init the block
                block["type"] = line[1:-1].rstrip()     
            else:
                key,value = line.split("=") 
                block[key.rstrip()] = value.lstrip()
        blocks.append(block)
        
        return blocks
    
    @classmethod
    def get_preprocessed_cfg(cls, cfgfile):
        with open(cfgfile, 'r') as file:
            lines = file.read().split('\n')                    # store the lines in a list
        lines = [x for x in lines if len(x) > 0]               # get rid of the empty lines 
        lines = [x for x in lines if x[0] != '#']              # get rid of comments
        lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
        return lines
    
    @classmethod
    def create_model_from_blocks(cls, blocks):
        outputs = []
        filters = []
        yolo_out_preds = []
        yolo_out_shapes = []
        
        net_info = blocks[0]
        input_shape = (int(net_info['width']), int(net_info['height']), int(net_info['channels']))
        batch_size = int(net_info['batch'])
        inputs = input_layer = Input(shape=input_shape, batch_size=batch_size)
        print("Input shape", inputs.shape)
        
        for i, block in enumerate(blocks[1:]):
            block_type = block['type']
            if block_type == 'convolutional':
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                padding = block['pad']
                activation = block['activation']
                
                try:
                    batch_normalize = int(block["batch_normalize"])
                    bias = False
                except:
                    batch_normalize = 0
                    bias = True
                    
                if padding:
                    pad = 'same'
                else:
                    pad = 'valid'
                    
                if stride > 1:
                    inputs = ZeroPadding2D(((1, 0), (1, 0)))(inputs)   
                    
                inputs = Conv2D(filters, kernel_size, strides=(stride, stride), 
                                padding=pad, activation=None, use_bias=bias,
                                name=F"conv_{i}")(inputs)
                
                if batch_normalize:
                    inputs = BatchNormalization(name=F"bnorm_{i}")(inputs)
                    
                if activation == 'leaky':
                    inputs = LeakyReLU(alpha=.1, name=F"leaky_{i}")(inputs)
                elif activation == 'linear':
                    # do nothing..
                    pass
                else:
                    raise Exception('Unknown activation function')
                                    
            elif block_type == 'upsample':
                stride = int(block['stride'])
                inputs = UpSampling2D(stride)(inputs)
            
            elif block_type == 'maxpool':
                size = int(block['size'])
                stride = int(block['stride'])
                
                inputs = MaxPool2D(pool_size=(size, size), strides=(stride, stride))(inputs)
                
            elif block_type == 'route':
                layers = block['layers'].split(',')
                layers = [int(l) for l in layers]
                
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                    
                if len(layers) == 1:    
                    #filters = output_filters[i + layers[0]]
                    inputs = outputs[i + layers[0]]
                elif len(layers) == 2:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    
                    # reorganize the previous layer 
                    target_shape = outputs[i + layers[1]].shape
                    outputs[i + layers[0]] = tf.reshape(outputs[i + layers[0]], 
                                [target_shape[0], target_shape[1], target_shape[2], -1])
                    #filters = output_filters[i + layers[0]] + output_filters[i + layers[1]] 
                    inputs = tf.concat([outputs[i + layers[0]], outputs[i + layers[1]]], axis=-1)
                else:
                    raise Exception('Invalid layers')
                
            elif block_type == 'shortcut':
                from_layer = int(block["from"])
                inputs = outputs[i - 1] + outputs[i + from_layer]
            
            elif block_type == 'yolo':
                num_classes = int(block['classes'])
                mask = block['mask'].split(',')
                mask = [int(m) for m in mask]
                anchors = block['anchors'].split(',')
                anchors = [int(a) for a in anchors]
                anchors = [(anchors[i], anchors[i + 1]) 
                               for i in range(0, len(anchors), 2)]
                anchors = [anchors[m] for m in mask]
                
                prediction = YoloLayer(anchors, num_classes, 
                                          input_layer.shape, name=F"yolo_{i}")(inputs)
                
                yolo_out_preds.append(prediction)
                yolo_out_shapes.append(prediction.shape)
            else:
                raise ValueError(F"Unexpected block type while parsing YOLO cfg file: {block_type}")
                    
            outputs.append(inputs)
            #output_filters.append(filters)
            print(outputs[-1])
            
        tf_model = tf.keras.Model(input_layer, yolo_out_preds)
        
        model = Model()
        model.tf_model = tf_model
        model.input_shape = input_shape
        model.yolo_output_shapes = yolo_out_shapes
        return model
    

def test_model_architecture():
    model = Model.from_cfg("cfg/yolov3-tiny.cfg")
    tf_model = model.tf_model
    print(tf_model.summary())
    
    

if __name__ == '__main__':
    test_model_architecture()   