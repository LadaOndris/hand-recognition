
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, \
    LeakyReLU, ZeroPadding2D, UpSampling2D
from tensorflow.keras import Model
print('Tensorflow Version', tf.__version__)


class Yolo(tf.keras.layers.Layer):
    def __init__(self, anchors, n_classes, input_layer_shape, name=None):
        super(Yolo, self).__init__(name=name)
        self.anchors = anchors
        self.n_anchors = len(anchors)
        self.n_classes = n_classes
        self.input_layer_shape = input_layer_shape
            
        
    def call(self, inputs):
        # transform to [None, B * grid size * grid size, 5 + C]
        # The B is the number of anchors and C is the number of classes.
        out_shape = inputs.get_shape().as_list()
        inputs = tf.reshape(inputs, [-1, self.n_anchors * out_shape[1] * out_shape[2], \
                                     5 + self.n_classes])
        
        # extract information
        box_centers = inputs[:, :, 0:2]
        box_shapes = inputs[:, :, 2:4]
        confidence = inputs[:, :, 4:5]
        classes = inputs[:, :, 5:self.n_classes + 5]
        
        # convert to range 0 - 1
        box_centers = tf.sigmoid(box_centers)
        confidence = tf.sigmoid(confidence)
        classes = tf.sigmoid(classes)
        
        # repeat anchors for all cells
        anchors = tf.tile(self.anchors, [out_shape[1] * out_shape[2], 1])
        box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)
        
        # create coordinates for each anchor for each cell
        # for 3 anchors per cell:
        # [0 0], [0 0], [0 0], [0 1], [0 1], [0 1], ...
        x = tf.range(out_shape[1], dtype=tf.float32)
        y = tf.range(out_shape[2], dtype=tf.float32)
        cx, cy = tf.meshgrid(x, y)
        cx = tf.reshape(cx, (-1, 1))
        cy = tf.reshape(cy, (-1, 1))
        cxy = tf.concat([cx, cy], axis=-1)
        cxy = tf.tile(cxy, [1, self.n_anchors])
        cxy = tf.reshape(cxy, [1, -1, 2])
        
        # get box_centers in real image coordinates
        strides = (self.input_layer_shape[1] // out_shape[1], \
                   self.input_layer_shape[2] // out_shape[2])
        box_centers = (box_centers + cxy) * strides
        
        # put it back together
        prediction = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)
        return prediction


def parse_cfg(cfgfile):
    lines = get_preprocessed_cfg(cfgfile)

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

def get_preprocessed_cfg(cfgfile):
    with open(cfgfile, 'r') as file:
        lines = file.read().split('\n')                    # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get rid of the empty lines 
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
    return lines

def create_model(blocks):
    
    outputs = []
    output_filters = []
    filters = []
    out_pred = []
    scale = 0
    
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
            
            prediction = Yolo(anchors, num_classes, 
                                      input_layer.shape, name=F"yolo_{i}")(inputs)
            
            if scale:
                out_pred = tf.concat([out_pred, prediction], axis=1)
            else:
                out_pred = prediction
                scale = 1
                
        outputs.append(inputs)
        #output_filters.append(filters)
        print(outputs[-1])
            
    model = Model(input_layer, out_pred)
    return model
    



def test_model_architecture():
    blocks = parse_cfg("cfg/yolov3-tiny.cfg")
    print("Blocks count:", len(blocks))
    print(create_model(blocks).summary())
    
    

if __name__ == '__main__':
    test_model_architecture()   