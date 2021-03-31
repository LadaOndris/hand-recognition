from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Dense, \
    MaxPooling2D, UpSampling2D
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
from src.utils.config import JGRJ2O_WEIGHT_DECAY
from src.utils.images import resize_bilinear_nearest_batch


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, kernel_size, filters):
        super(ResnetBlock, self).__init__(name='')
        self.kernel_size = kernel_size
        self.filters = filters
        half_filters = int(filters // 2)

        self.conv2d_1 = conv(half_filters, (1, 1))
        self.bn_1 = bn()
        self.conv2d_2 = conv(half_filters, kernel_size, padding='same')
        self.bn_2 = bn()
        self.conv2d_3 = conv(filters, (1, 1))
        self.bn_3 = bn()
        self.conv2d_4 = conv(filters, (1, 1))
        self.bn_4 = bn()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'kernel_size': self.kernel_size,
            'filters': self.filters,
        })
        return config

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


def conv(filters, kernel_size, strides=(1, 1), padding='valid', use_bias=True):
    initializer = tf.keras.initializers.TruncatedNormal(0.0, 0.01)
    regularizer = tf.keras.regularizers.L1L2(l2=JGRJ2O_WEIGHT_DECAY)
    return Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=initializer,
                  kernel_regularizer=regularizer, use_bias=use_bias)


def bn():
    initializer = tf.keras.initializers.TruncatedNormal(1.0, 0.01)
    regularizer = tf.keras.regularizers.L1L2(l2=JGRJ2O_WEIGHT_DECAY)
    return BatchNormalization(gamma_initializer=initializer, gamma_regularizer=regularizer)


def conv_bn(x, filters, kernel_size, strides=(1, 1), padding='valid'):
    x = conv(filters, kernel_size, strides=strides, padding=padding)(x)
    x = bn()(x)
    return x


def conv_bn_relu(x, filters, kernel_size, strides=(1, 1), padding='valid'):
    x = conv(filters, kernel_size, strides=strides, padding=padding)(x)
    x = bn()(x)
    return ReLU()(x)


class JGR_J2O:

    def __init__(self, input_size=96, n_joints=21, n_features=128):
        self.input_size = input_size
        self.out_size = input_size // 4
        self.n_joints = n_joints
        self.n_features = n_features
        self.A_e = self.connection_weight_matrix()

    def connection_weight_matrix(self):
        # This is A + I:
        A = np.array([[1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                      [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
        D = np.diag(np.power(np.sum(A, axis=0), -0.5))
        A_e = np.dot(np.dot(D, A), D)
        A_e = tf.constant(A_e, dtype=tf.float32)
        return A_e

    def hourglass_module(self, inputs, n, features):
        with tf.name_scope(name='hourglass'):
            left = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
            left = ResnetBlock(kernel_size=(3, 3), filters=features)(left)
            top = ResnetBlock(kernel_size=(3, 3), filters=features)(inputs)

            if n > 1:
                middle = self.hourglass_module(left, n - 1, features)
            else:
                middle = left

            right = ResnetBlock(kernel_size=(3, 3), filters=features)(middle)
            right = UpSampling2D(size=(2, 2), interpolation='nearest')(right)
            return right + top

    def joint_graph_reasoning_module(self, x):
        w, F = self.pixel_to_joint_voting(x)  # produces joints' features
        Fe = self.graph_reasoning(F)
        newFe = self.joint_to_pixel_mapping(Fe, w)

        # local feature enhancement
        x_ = tf.concat([newFe, x], axis=-1)
        x_ = conv_bn_relu(x_, self.n_features, (1, 1))
        return x_, w

    def pixel_to_joint_voting(self, x):
        weights = conv(self.n_joints, (1, 1))(x)
        weights = self.spatial_softmax(weights)  # [-1, N, W * H]

        x = conv_bn_relu(x, x.shape[-1], (1, 1))
        x = tf.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[3]])  # [-1, W * H, features (C)]
        F = tf.matmul(weights, x)
        w_reshaped = tf.reshape(weights, [-1, self.n_joints, self.out_size, self.out_size])
        return w_reshaped, F

    def graph_reasoning(self, F):
        """
        Augments joints' feature representations
        by computing the following matrix multiplication
        F_e = σ(A_e @ F @ W_e), where
        A_e is a connection weight matrix defining joint dependencies,
        W_e is a trainable transformation matrix,
        σ is a nonlinear function.

        Parameters
        ----------
        F
            Joints' feature representations
        Returns
        ----------
            Returns augmented joints' feature representations
            of the same shape as F.
        """
        # Matrix multiplication through trainable matrix W_e
        F_reshaped = tf.reshape(F, [-1, self.n_features])
        FWe = Dense(F.shape[-1], use_bias=False)(F_reshaped)
        FWe = tf.reshape(FWe, [-1, self.n_joints, self.n_features])
        F_augmented = tf.matmul(self.A_e, FWe)
        F_augmented = ReLU()(F_augmented)
        return F_augmented

    def joint_to_pixel_mapping(self, Fe, w):
        """
        (-1, 21, 128) -> (-1, 24, 24, 128)

        Parameters
        ----------
        Fe
        w

        Returns
        -------

        """
        # (-1, 21, 128) -> (-1, 21, 1, 1, 128)
        newFe = Fe[:, :, tf.newaxis, tf.newaxis, :]
        # (-1, 21, 1, 1, 128) -> (-1, 21, 24, 24, 128)
        newFe = tf.tile(newFe, [1, 1, self.out_size, self.out_size, 1])
        # (-1, 21, 24, 24, 128) -> (-1, 24, 24, 128)
        newFe = newFe * w[..., tf.newaxis]  # (-1, 21, 24, 24, 128)
        newFe = tf.reduce_mean(newFe, axis=1)  # finally (-1, 24, 24, 128)
        newFe = conv_bn_relu(newFe, self.n_features, kernel_size=(1, 1))
        return newFe

    def spatial_softmax(self, features):
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

    def pixel_to_offset_module(self, x):
        u_offsets = conv(self.n_joints, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(x)
        v_offsets = conv(self.n_joints, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(x)
        z_offsets = conv(self.n_joints, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(x)
        return u_offsets, v_offsets, z_offsets

    def graph(self):
        input = Input(shape=(self.input_size, self.input_size, 1))

        # The following layers precede the hourglass module
        # according to Hourglass and JGR-P2O papers
        x = conv_bn_relu(input, filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same')  # 48, 48, 32
        x = ResnetBlock(kernel_size=(3, 3), filters=64)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = ResnetBlock(kernel_size=(3, 3), filters=x.shape[-1])(x)
        x = ResnetBlock(kernel_size=(3, 3), filters=self.n_features)(x)  # 24, 24, hourglass_features

        # The number of features stays the same across the whole hourglass module
        x = self.hourglass_module(x, n=3, features=x.shape[-1])
        x = ResnetBlock(kernel_size=(3, 3), filters=x.shape[-1])(x)

        x_jgr, weights = self.joint_graph_reasoning_module(x)
        u_offs, v_offs, z_offs = self.pixel_to_offset_module(x_jgr)
        offsets = tf.stack([u_offs, v_offs, z_offs], axis=-1, name='offsets')
        # offs.shape [-1, 24, 24, 21]
        # u_im, v_im, z_im

        weights = tf.transpose(weights, [0, 2, 3, 1])  # [-1, 24, 24, 21]

        # u_im.shape [-1, 24, 24, 21]
        x = tf.range(self.out_size)
        y = tf.range(self.out_size)
        x, y = tf.meshgrid(x, y)
        # expand_dims, cast, and normalize to [0, 1]
        u_im = tf.cast(x[:, :, tf.newaxis], tf.float32) / self.out_size
        v_im = tf.cast(y[:, :, tf.newaxis], tf.float32) / self.out_size
        # Z coordinate is retrieved from the image directly
        # (values should be already normalized to [-1, 1]:
        # z_im = tf.image.resize(input, [self.out_size, self.out_size],
        #                        method=tf.image.ResizeMethod.BILINEAR)
        z_im = resize_bilinear_nearest_batch(input, [self.out_size, self.out_size])

        # u, v, z: [-1, 21]
        u = tf.reduce_sum(weights * (u_im + u_offs), axis=[1, 2])
        v = tf.reduce_sum(weights * (v_im + v_offs), axis=[1, 2])
        z = tf.reduce_sum(weights * (z_im + z_offs), axis=[1, 2])
        uvz = tf.stack([u, v, z], axis=-1, name='joints')

        return Model(input, outputs=[uvz, offsets])
