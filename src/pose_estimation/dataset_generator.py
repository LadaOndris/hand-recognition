import tensorflow as tf
import numpy as np


class DatasetGenerator:
    """
    DatasetGenerator takes an iterator of a hand pose dataset
    as its input, and preprocesses the joints coordinates
    to match the expected output of JGR-J2O network.
    """

    def __init__(self, dataset_iterator, image_out_size):
        self.iterator = dataset_iterator
        self.image_out_size = image_out_size

    def __iter__(self):
        return self

    def __next__(self):
        # The dataset iterator should return images of size (96, 96),
        # which is defined as a parameter to the JGR-J2O network.
        # The depth image should be already normalized to [-1, 1],
        # as well as the UV coords to [0, 1], and Z coord to [-1, 1].
        batch_images, batch_joints = self.iterator.get_next()
        offsets = self.compute_offsets(batch_joints)
        y_true = {'coords': batch_joints, 'offsets': offsets}
        return batch_images, y_true

    def compute_offsets(self, joints):
        """
            Computes offsets for each joint coordinate.
            Offsets are normalized to [-1, 1] if the UVZ coords are
            normalized to [0, 1].
        Parameters
        ----------
        joints  tf.Tensor of shape [None, n_joints, 3]

        Returns
        -------
            tf.Tensor of shape [None, out_size, out_size, n_joints, 3]
        """
        n_joints = joints.shape[1]
        x = self.offset_coord(joints[..., 0], n_joints)  # shape = [None, out_size, n_joints]
        y = self.offset_coord(joints[..., 1], n_joints)  # shape = [None, out_size, n_joints]

        u_offsets = y[:, tf.newaxis, :, :]
        u_offsets = tf.tile(u_offsets, [1, self.image_out_size, 1, 1])

        v_offsets = y[:, :, tf.newaxis, :]
        v_offsets = tf.tile(v_offsets, [1, 1, self.image_out_size, 1])

        z_coords = joints[..., 2]
        z_coords = z_coords[:, tf.newaxis, tf.newaxis, :]
        z_offsets = tf.tile(z_coords, [1, self.image_out_size, self.image_out_size, 1])
        return tf.stack([u_offsets, v_offsets, z_offsets], axis=-1)

    def offset_coord(self, joints_single_coord, n_joints):
        x = tf.linspace(0, 1, num=self.image_out_size)  # [0, 1]
        x = tf.cast(x, tf.float32)
        x = x[tf.newaxis, :, tf.newaxis]  # shape = [1, out_size, 1]
        x = tf.tile(x, [1, 1, n_joints])  # shape = [1, out_size, n_joints]
        x -= tf.expand_dims(joints_single_coord, 1)  # [0 - u, 1 - u], shape = [None, out_size, n_joints]
        x *= -1  # [u, u - 1]
        return x


if __name__ == "__main__":
    gen = DatasetGenerator(None, 5)
    joints = np.full([4, 21, 3], 0.6, dtype=np.float32)
    joints = tf.constant(joints, dtype=tf.float32)
    offsets = gen.compute_offsets(joints)
    pass
