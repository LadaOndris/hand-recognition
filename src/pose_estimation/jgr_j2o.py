from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, GlobalAveragePooling2D, Dense, \
    MaxPooling2D, UpSampling2D
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
from src.pose_estimation.loss import CoordinateAndOffsetLoss


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


class JGR_J2O:

    def __init__(self):
        self.n_joints = 21
        self.n_features = 128
        self.A_e = self.connection_weight_matrix()
        pass

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
        with tf.name_scope(name='hourglass') as scope:
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
        x_ = self.conv_bn_relu(x_, self.n_features, (1, 1))
        return x_, w

    def conv_bn_relu(self, x, filters, kernel_size):
        x = Conv2D(filters, kernel_size)(x)
        x = BatchNormalization()(x)
        return ReLU()(x)

    def pixel_to_joint_voting(self, x):
        weights = Conv2D(self.n_joints, (1, 1))(x)
        weights = self.spatial_softmax(weights)  # [-1, W, dim * dim]

        x = self.conv_bn_relu(x, x.shape[-1], (1, 1))
        x = tf.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[3]])  # [-1, features, dim * dim]
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
        newFe = self.conv_bn_relu(newFe, self.n_features, kernel_size=(1, 1))
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
        u_offsets = Conv2D(self.n_joints, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(x)
        v_offsets = Conv2D(self.n_joints, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(x)
        z_offsets = Conv2D(self.n_joints, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(x)
        return u_offsets, v_offsets, z_offsets

    def graph(self):
        self.input_size = 96
        self.out_size = self.input_size // 4
        input = Input(shape=(self.input_size, self.input_size, 1))

        # The following layers precede the hourglass module
        # according to Hourglass and JGR-P2O papers
        x = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same',
                   kernel_initializer='normal')(input)  # 48, 48, 32
        x = ResnetBlock(kernel_size=(3, 3), filters=64)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = ResnetBlock(kernel_size=(3, 3), filters=x.shape[-1])(x)
        x = ResnetBlock(kernel_size=(3, 3), filters=self.n_features)(x)  # 24, 24, hourglass_features

        # The number of features stays the same across the whole hourglass module
        x = self.hourglass_module(x, n=3, features=x.shape[-1])
        x = ResnetBlock(kernel_size=(3, 3), filters=x.shape[-1])(x)

        x_jgr, weights = self.joint_graph_reasoning_module(x)
        u_offs, v_offs, z_offs = self.pixel_to_offset_module(x_jgr)
        offsets = tf.stack([u_offs, v_offs, z_offs], axis=-1)
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
        z_im = tf.image.resize(input, [self.out_size, self.out_size],
                               method=tf.image.ResizeMethod.BILINEAR)

        # u, v, z: [-1, 21]
        u = tf.reduce_sum(weights * (u_im + u_offs), axis=[1, 2])
        v = tf.reduce_sum(weights * (v_im + v_offs), axis=[1, 2])
        z = tf.reduce_sum(weights * (z_im + z_offs), axis=[1, 2])
        uvz = tf.stack([u, v, z], axis=-1)

        outputs = {'coords': uvz, 'offsets': offsets}
        return Model(input, outputs=[outputs])

def train():
    network = JGR_J2O()
    model = network.graph()
    print(model.summary())

    adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.96)
    loss = CoordinateAndOffsetLoss()
    model.compile(optimizer=adam, loss=loss)

from src.datasets.bighand.dataset import BighandDataset
from src.pose_estimation.dataset_generator import DatasetGenerator
from src.utils.paths import BIGHAND_DATASET_DIR

def evaluate():
    # load model!
    network = JGR_J2O()
    model = network.graph()

    # initialize dataset
    im_out_size = 24
    bighand_ds = BighandDataset(BIGHAND_DATASET_DIR, train_size=0.9, batch_size=1, shuffle=False)
    gen = DatasetGenerator(iter(bighand_ds.test_dataset), im_out_size)

    for batch in gen:
        images, y_true = batch
        model.predict()

