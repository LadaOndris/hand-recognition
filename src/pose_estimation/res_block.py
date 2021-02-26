from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, GlobalAveragePooling2D, Dense, \
    MaxPooling2D, UpSampling2D
from tensorflow.keras import Model
import tensorflow as tf


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, kernel_size, filters):
        super(ResnetBlock, self).__init__(name='')
        self.filters = filters
        half_filters = int(filters // 2)

        self.conv2d_1 = tf.keras.layers.Conv2D(half_filters, (1, 1))
        self.bn_1 = tf.keras.layers.BatchNormalization()

        self.conv2d_2 = tf.keras.layers.Conv2D(half_filters, kernel_size, padding='same')
        self.bn_2 = tf.keras.layers.BatchNormalization()

        self.conv2d_3 = tf.keras.layers.Conv2D(filters, (1, 1))
        self.bn_3 = tf.keras.layers.BatchNormalization()

        self.conv2d_4 = tf.keras.layers.Conv2D(filters, (1, 1))
        self.bn_4 = tf.keras.layers.BatchNormalization()

    def call(self, input, training=False):
        x = self.conv2d_1(input)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2d_2(x)
        x = self.bn_2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2d_3(x)
        x = self.bn_3(x, training=training)

        if input.shape[-1] != self.filters:
            input = self.conv2d_4(input)
            input = self.bn_4(input, training=training)

        x += input
        return tf.nn.relu(x)


# TODO
class HourglassModule(tf.keras.layers.Layer):
    def __init__(self, n, input_shape):
        super(HourglassModule, self).__init__(name='')
        self.features = input_shape[-1]
        self.x = Input(shape=input_shape[1:])
        self.graph = self.create_recursively(self.x, n)

    def create_recursively(self, input, n):
        left = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input)
        left = ResnetBlock(kernel_size=(3, 3), filters=self.features)(left)
        top = ResnetBlock(kernel_size=(3, 3), filters=self.features)(input)

        if n > 1:
            middle = self.create_recursively(left, n - 1)
        else:
            middle = left

        right = ResnetBlock(kernel_size=(3, 3), filters=self.features)(middle)
        right = UpSampling2D(size=(2, 2), interpolation='nearest')(right)
        return right + top

    def call(self, inputs, training=False):
        return Model(self.x, self.graph)(inputs)


def hourglass_module(inputs, n, features):
    with tf.name_scope(name='hourglass') as scope:
        left = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
        left = ResnetBlock(kernel_size=(3, 3), filters=features)(left)
        top = ResnetBlock(kernel_size=(3, 3), filters=features)(inputs)

        if n > 1:
            middle = hourglass_module(left, n - 1, features)
        else:
            middle = left

        right = ResnetBlock(kernel_size=(3, 3), filters=features)(middle)
        right = UpSampling2D(size=(2, 2), interpolation='nearest')(right)
        return right + top


def joint_graph_reasoning_module(x):
    w, F = pixel_to_joint_voting(x)
    Fe = graph_reasoning(F)
    x_ = joint_to_pixel_mapping(Fe, w)

    # local feature enhancement
    x_ = x_ + x
    x_ = conv_bn_relu(x_, x.shape[-1], (1, 1))
    return x_


def conv_bn_relu(x, filters, kernel_size):
    x = tf.keras.layers.Conv2D(filters, kernel_size)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)


def pixel_to_joint_voting(x):
    weights = tf.keras.layers.Conv2D(n_joints, (1, 1))(x)
    weights = spatial_softmax(weights)  # [-1, W, dim * dim]

    x = conv_bn_relu(x, x.shape[-1], (1, 1))
    x = tf.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[3]])  # [-1, features, dim * dim]
    F = tf.matmul(weights, x)
    return weights, F

def graph_reasoning(F):
    pass

def joint_to_pixel_mapping(Fe, w):
    pass

def spatial_softmax(features):
    """
    Computes the softmax function for four-dimensional array.
    Parameters
    ----------
    features
        Features has a shape (batch_size, height, width, channels).
    """
    _, H, W, C = features.shape
    features = tf.reshape(features, [-1, H * W, C])
    features = tf.transpose(features, [0, 2, 1])
    # features = tf.reshape(tf.transpose(features, [0, 3, 1, 2]), [B * C, H * W])
    softmax = tf.nn.softmax(features, axis=1)
    # softmax = tf.reshape(softmax, [B, C, H, W])
    # softmax = tf.transpose(tf.reshape(softmax, [B, C, H, W]), [0, 2, 3, 1])
    return softmax


# TODO
class PixelToOffsetModule(tf.keras.layers.Layer):
    def __init__(self):
        super(PixelToOffsetModule, self).__init__()

    def call(self, inputs):
        return inputs


n_joints = 21
hourglass_features = 128
input = Input(shape=(96, 96, 1))

# The following layers precede the hourglass module
# according to Hourglass and JGR-P2O papers
x = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same',
           kernel_initializer='normal')(input)  # 48, 48, 32
x = ResnetBlock(kernel_size=(3, 3), filters=64)(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
x = ResnetBlock(kernel_size=(3, 3), filters=x.shape[-1])(x)
x = ResnetBlock(kernel_size=(3, 3), filters=hourglass_features)(x)  # 24, 24, hourglass_features

# The number of features stays the same across the whole hourglass module
x = hourglass_module(x, n=3, features=x.shape[-1])
x = ResnetBlock(kernel_size=(3, 3), filters=x.shape[-1])(x)

x_jgr = joint_graph_reasoning_module(x)
x_p2o = PixelToOffsetModule()(x)

model = Model(input, x)

print(model.summary())
