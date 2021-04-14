import tensorflow as tf


class MeanJointErrorMetric(tf.keras.metrics.Metric):

    def __init__(self, name='mean_joint_error', **kwargs):
        super(MeanJointErrorMetric, self).__init__(name=name, **kwargs)
        self._total = self.add_weight(name='total', initializer='zeros')
        self._count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, xyz_true, xyz_pred, sample_weight=None):
        mje = self.mean_joint_error(xyz_true, xyz_pred)
        self._total.assign_add(tf.cast(mje, dtype=tf.float32))
        self._count.assign_add(1)

    def mean_joint_error(self, joints1, joints2):
        distances = tf.norm(joints1 - joints2, axis=2)
        return tf.reduce_mean(distances)

    def result(self):
        return tf.math.divide_no_nan(self._total, self._count)

    def reset_states(self):
        self._total.assign(0.)
        self._count.assign(0.)


class DistancesBelowThreshold(tf.keras.metrics.Metric):

    def __init__(self, max_thres=100, name="distance_below_threshold", **kwargs):
        super(DistancesBelowThreshold, self).__init__(name=name, **kwargs)
        self.max_thres = max_thres
        self.max_distances = None

    def update_state(self, xyz_true, xyz_pred):
        """
        Updates current state of this metric,
        given a batch of true and predicted coordinates.
        Coordinate parameters have shape (None, 21, 3).
        """
        distances = tf.norm(xyz_true - xyz_pred, axis=2)
        max_distances = tf.reduce_max(distances, axis=1)
        if self.max_distances is None:
            self.max_distances = max_distances
        else:
            self.max_distances = tf.concat([self.max_distances, max_distances], axis=0)

    def result(self):
        allowed_distances = tf.range(1, self.max_thres + 1, dtype=tf.float32)
        allowed_distances = allowed_distances[:, tf.newaxis]
        num_samples = self.max_distances.shape[0]
        allowed_distances = tf.tile(allowed_distances, [1, num_samples])
        distances = self.max_distances[tf.newaxis, :]
        mask = tf.where(distances < allowed_distances, 1, 0)
        counts = tf.reduce_sum(mask, axis=-1)
        proportions = counts / num_samples
        return proportions

    def reset_states(self):
        self.max_distances = None