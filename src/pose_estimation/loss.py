import tensorflow as tf
import numpy as np


class CoordinateAndOffsetLoss(tf.keras.losses.Loss):

    def __init__(self, balance=0.0001, name='coordinate_and_offset_loss', *args, **kwargs):
        super(CoordinateAndOffsetLoss, self).__init__(name=name, *args, **kwargs)
        # self.offset_layer = model.get_layer('offsets')
        self.balance = balance
        self.huber_loss = tf.keras.losses.Huber()

    def call(self, y_true, y_pred):
        # y.shape: [-1, 21, 3]
        l_coord = self.loss_coordinate(y_true['coords'], y_pred['coords'])
        l_offset = self.loss_offset(y_true['offsets'], y_pred['offsets'])
        return l_coord + self.balance * l_offset

    def loss_coordinate(self, y_true, y_pred):
        return self.huber_loss(y_true, y_pred)

    def loss_offset(self, y_true, y_pred):
        return self.huber_loss(y_true, y_pred)
