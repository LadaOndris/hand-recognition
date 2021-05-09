import tensorflow as tf


class CoordinateAndOffsetLoss(tf.keras.losses.Loss):

    def __init__(self, balance=0.0001, name='coordinate_and_offset_loss', *args, **kwargs):
        super(CoordinateAndOffsetLoss, self).__init__(name=name, *args, **kwargs)
        # self.offset_layer = model.get_layer('offsets')
        self.balance = balance
        self.huber_loss = tf.keras.losses.Huber()

    def call(self, y_true, y_pred):
        # y.shape: [-1, 21, 3]
        return self.huber_loss(y_true, y_pred)

    def loss_coordinate(self, y_true, y_pred):
        return self.huber_loss(y_true, y_pred)

    def loss_offset(self, y_true, y_pred):
        return self.huber_loss(y_true, y_pred)


class CoordinateLoss(tf.keras.losses.Loss):

    def __init__(self, name='coordinate_loss', *args, **kwargs):
        super(CoordinateLoss, self).__init__(name=name, *args, **kwargs)
        self.huber_loss = tf.keras.losses.Huber()

    def call(self, y_true, y_pred):
        # tf.print('coords:', tf.shape(y_pred))
        return self.huber_loss(y_true, y_pred)


class OffsetLoss(tf.keras.losses.Loss):

    def __init__(self, balance=0.0001, name='offset_loss', *args, **kwargs):
        super(OffsetLoss, self).__init__(name=name, *args, **kwargs)
        self.huber_loss = tf.keras.losses.Huber()
        self.balance = balance

    def call(self, y_true, y_pred):
        # tf.print('offsets:', tf.shape(y_pred))
        return self.balance * self.huber_loss(y_true, y_pred)
