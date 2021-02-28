import tensorflow as tf

class MeanJointErrorMetric(tf.keras.metrics.Metric):

    def __init__(self, name='mean_joint_error', **kwargs):
        super(MeanJointErrorMetric, self).__init__(name=name, **kwargs)
        self._total = self.add_weight(name='total', initializer='zeros')
        self._count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        mje = self.mean_joint_error(y_true, y_pred)
        self._total.assign_add(tf.cast(mje, dtype=tf.float32))
        self._count.assign_add(1)

    def mean_joint_error(self, joints1, joints2):
        diff = tf.norm(joints1 - joints2)
        return tf.reduce_mean(diff)

    def result(self):
        return tf.math.divide_no_nan(self._total, self._count)

    def reset_states(self):
        self._total.assign(0.)
        self._count.assign(0.)