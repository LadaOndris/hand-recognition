from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, GlobalAveragePooling2D, Dense, \
    MaxPooling2D
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
    def __init__(self, kernel_size, filters):
        super(HourglassModule, self).__init__(name='')

    def call(self, inputs, training=False):
        return inputs


# TODO
class JointGraphReasoningModule(tf.keras.layers.Layer):
    def __init__(self):
        super(JointGraphReasoningModule, self).__init()

    def call(self, inputs):
        return inputs


# TODO
class PixelToOffsetModule(tf.keras.layers.Layer):
    def __init__(self):
        super(PixelToOffsetModule, self).__init()

    def call(self, inputs):
        return inputs


hourglass_features = 128
input = Input(shape=(96, 96, 1))
x = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same',
           kernel_initializer='normal')(input)  # 48, 48, 32
x = ResnetBlock(kernel_size=(3, 3), filters=64)(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
x = ResnetBlock(kernel_size=(3, 3), filters=x.shape[-1])(x)
x = ResnetBlock(kernel_size=(3, 3), filters=hourglass_features)(x)  # 24, 24, hourglass_features

x = HourglassModule()(x)
x = ResnetBlock(kernel_size=(3, 3), filters=x.shape[-1])(x)

x_jgr = JointGraphReasoningModule()(x)
x_p2o = PixelToOffsetModule()(x)

model = Model(input, x)

print(model.summary())
