import tensorflow as tf
import numpy as np


class DatasetGenerator:
    """
    DatasetGenerator takes an iterator of a hand pose dataset
    as its input, and preprocesses the joints coordinates
    to match the expected output of JGR-J2O network.
    """

    def __init__(self, dataset_iterator, image_in_size, image_out_size, return_xyz=False):
        self.iterator = dataset_iterator
        self.image_in_size = image_in_size
        self.image_out_size = image_out_size
        self.return_xyz = return_xyz

    def __iter__(self):
        return self

    def __next__(self):
        # The dataset iterator should return images of size (96, 96),
        # which is defined as a parameter to the JGR-J2O network.
        # The depth image should be already normalized to [-1, 1],
        # as well as the UV coords to [0, 1], and Z coord to [-1, 1].
        images, joints, bboxes = self.iterator.get_next()
        # crop image, resize to fixed size

        invert_resize_coeffs = []
        for i, bbox in enumerate(bboxes):
            cropped_image = images[i, bboxes[i, 0]:bboxes[i, 2], bboxes[i, 1]:bboxes[i, 3]]
            resized_image = tf.image.resize(cropped_image, [self.image_in_size, self.image_in_size])
            # save the resize coeffs to revert the resizing
            invert_resize_coeffs.append(cropped_image.shape / self.image_in_size)
            # should i resize the coords???
        # normalize image values to range [0, 1]

        # normalize coordinates

        uvz_joints = self.preprocess_joints()
        offsets = self.compute_offsets(joints)
        y_true = {'coords': joints, 'offsets': offsets}
        return images, y_true

    def preprocess_joints(self, joints):
        """
        Converts XYZ joints to UVZ coordinates.
        """

    def compute_offsets(self, joints):
        """
            Computes offsets for each joint coordinate.
            Offsets are normalized to [-1, 1] if the given
            UVZ coords are normalized to [0, 1].
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
