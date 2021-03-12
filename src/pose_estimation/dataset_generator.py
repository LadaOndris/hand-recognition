import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from src.utils.camera import Camera
from src.utils.plots import plot_joints_2d


class DatasetGenerator:
    """
    DatasetGenerator takes an iterator of a hand pose dataset
    as its input, and preprocesses the joints coordinates
    to match the expected output of JGR-J2O network.
    """

    def __init__(self, dataset_iterator, depth_image_size, image_in_size, image_out_size, camera: Camera,
                 return_xyz=False, dataset_includes_bboxes=False, augment=False):
        """
        Parameters
        ----------
        dataset_iterator
        image_in_size
        image_out_size
        camera
        return_xyz
        dataset_includes_bboxes
            If dataset returns bboxes, then the images are expected to be already cropped.
        """
        self.iterator = dataset_iterator
        self.depth_image_size = depth_image_size
        self.image_in_size = image_in_size
        self.image_out_size = image_out_size
        self.camera = camera
        self.return_xyz = return_xyz
        self.dataset_includes_bboxes = dataset_includes_bboxes
        self.augment = augment
        self.max_depth = 2048
        self.bboxes = None  # These are used to position the cropped coordinates into global picture
        self.resize_coeffs = None  # These are used to invert image resizing
        self.cropped_images = None

    def __iter__(self):
        return self

    def postprocess(self, y_pred):
        """
        Denormalizes predicted coordinates.
        Applies inverted resizing.
        Puts the coordinates in the global picture by
        adding bounding box offsets.

        Returns
        -------
            Global UVZ coordinates.
        """
        normalized_uvz = y_pred[0]
        normalized_uv = normalized_uvz[..., :2]
        normalized_z = normalized_uvz[..., 2:3]

        resized_uv = normalized_uv * [self.image_in_size, self.image_in_size]
        global_z = normalized_z * self.max_depth  # [:, tf.newaxis, tf.newaxis] TODO
        uv_local = 1.0 / self.resize_coeffs[:, tf.newaxis, :] * resized_uv
        uv_global = uv_local + tf.cast(self.bboxes[:, tf.newaxis, :2], dtype=tf.float32)
        uvz_global = tf.concat([uv_global, global_z], axis=-1)
        return uvz_global

    def __next__(self):
        """
        The dataset iterator returns images of arbitrary size -
        either original size, e.g. 640x480 (BigHand dataset), or cropped images using
        the returned bounding boxes (MSRA dataset).
        These images are cropped if no bounding boxes are returned from
        the iterator.
        Then, they are resized to shape (96, 96), which is the
        input shape of the JGR-J20 network.
        The dataset iterator should return joint coordinates in global
        coordinate system (XYZ). They are also cropped, resized.
        Both, images and coordinates are normalized to range [0, 1].
        Also, offsets are computed from the normalized coordinates.
        """
        if self.dataset_includes_bboxes:
            images, ignore_bboxes, xyz_global = self.iterator.get_next()
        else:
            images, xyz_global = self.iterator.get_next()
        uv_global = self.camera.world_to_pixel(xyz_global)

        if self.augment:
            images, uv_global = self.augment_batch(images, uv_global)

        self.bboxes = self.extract_bboxes(uv_global)
        self.bboxes = self.square_bboxes(self.bboxes)
        cropped_imgs = self.crop_images(images, self.bboxes)
        self.xyz_global = xyz_global
        self.cropped_images = cropped_imgs
        uv_local = uv_global - tf.cast(self.bboxes[:, tf.newaxis, :2], dtype=tf.float32)

        resize_new_size = [self.image_in_size, self.image_in_size]

        def _resize(img):
            return tf.image.resize(img.to_tensor(), resize_new_size)

        resized_imgs = tf.map_fn(_resize, cropped_imgs,
                                 fn_output_signature=tf.TensorSpec(shape=(self.image_in_size, self.image_in_size, 1),
                                                                   dtype=tf.float32))
        cropped_imgs_sizes_u = self.bboxes[..., 2] - self.bboxes[..., 0]  # right - left
        cropped_imgs_sizes_v = self.bboxes[..., 3] - self.bboxes[..., 1]  # bottom - top
        cropped_imgs_sizes = tf.stack([cropped_imgs_sizes_u, cropped_imgs_sizes_v], axis=-1)
        self.resize_coeffs = tf.cast(resize_new_size / cropped_imgs_sizes, dtype=tf.float32)
        resized_uv = self.resize_coeffs[:, tf.newaxis, :] * uv_local

        self.normalized_imgs, normalized_uvz = self.normalize(resized_imgs, resized_uv, xyz_global[..., 2:3])
        offsets = self.compute_offsets(self.normalized_imgs, normalized_uvz)
        return self.normalized_imgs, [normalized_uvz, offsets]

    def normalize(self, images, joints_uv, joints_z):
        # self.max_depth = tf.math.reduce_max(joints_z, axis=[1, 2])
        normalized_uv = joints_uv / [self.image_in_size, self.image_in_size]
        normalized_z = joints_z / self.max_depth  # [:, tf.newaxis, tf.newaxis]
        normalized_uvz = tf.concat([normalized_uv, normalized_z], axis=-1)

        normalized_imgs = images / self.max_depth  # [:, tf.newaxis, tf.newaxis, tf.newaxis]
        normalized_imgs = tf.where(normalized_imgs > 1, 0, normalized_imgs)
        return normalized_imgs, normalized_uvz

    def augment_batch(self, images, uv_joints):
        images = tf.cast(images, dtype=tf.float32)
        im_height, im_width = images.shape[1], images.shape[2]
        image_center = [im_width / 2, im_height / 2]
        images, joints = self.rotate(images, uv_joints, image_center)
        return images, joints

    def rotate(self, images, uv_joints, image_center):
        degrees = tf.random.uniform(shape=[tf.shape(images)[0]], minval=-180, maxval=180)
        radians = np.deg2rad(degrees)
        # Transposed rotation matrix
        rotation_matrix = np.array([[np.cos(radians), np.sin(radians)],
                                    [-np.sin(radians), np.cos(radians)]])
        new_images = tfa.image.rotate(images, radians)
        # Set the image center at [96/2, 96/2] and rotate around the center
        uv_joints = uv_joints - image_center
        new_joints = tf.matmul(tf.transpose(rotation_matrix, [2, 0, 1]), tf.transpose(uv_joints, [0, 2, 1]))
        new_joints = tf.transpose(new_joints, [0, 2, 1])
        # Set the center at [0, 0] of the image
        new_joints = new_joints + image_center
        return new_images, new_joints

    def extract_bboxes(self, uv_global):
        """
        Parameters
        ----------
        images      tf.Tensor of shape [None, height, width]
        uv_global   tf.Tensor of shape [None, n_joints, 2]

        Returns
        -------
            tf.Tensor of shape [None, 4].
            The four values are [left, top, right, bottom].
        """
        # Extract u and v coordinates
        u = uv_global[..., 0]
        v = uv_global[..., 1]

        # Find min and max over joints axis.
        u_min, u_max = tf.reduce_min(u, axis=1), tf.reduce_max(u, axis=1)
        v_min, v_max = tf.reduce_min(v, axis=1), tf.reduce_max(v, axis=1)

        # Move min and max to make the bbox slighty larger
        shift_coeff = 0.2
        width = u_max - u_min
        height = v_max - v_min
        u_shift = width * shift_coeff
        v_shift = height * shift_coeff
        u_min, u_max = u_min - u_shift, u_max + u_shift
        v_min, v_max = v_min - v_shift, v_max + v_shift

        # The bounding box is represented by four coordinates
        bboxes = tf.stack([u_min, v_min, u_max, v_max], axis=-1)
        bboxes = tf.cast(bboxes, dtype=tf.int32)
        return bboxes

    def square_bboxes(self, bboxes):
        u_min, v_min, u_max, v_max = tf.unstack(bboxes, axis=-1)
        width = u_max - u_min
        height = v_max - v_min

        # Make bbox square in the U axis
        u_min_new = u_min + tf.cast(width / 2 - height / 2, dtype=u_min.dtype)
        u_max_new = u_min + tf.cast(width / 2 + height / 2, dtype=u_min.dtype)
        u_min_raw = tf.where(width < height, u_min_new, u_min)
        u_max_raw = tf.where(width < height, u_max_new, u_max)
        u_min = tf.math.maximum(u_min_raw, 0)
        u_max = tf.math.minimum(u_max_raw, self.depth_image_size[0])

        # Make bbox square in the V axis
        v_min_new = v_min + tf.cast(height / 2 - width / 2, dtype=v_min.dtype)
        v_max_new = v_min + tf.cast(height / 2 + width / 2, dtype=v_min.dtype)
        v_min_raw = tf.where(height < width, v_min_new, v_min)
        v_max_raw = tf.where(height < width, v_max_new, v_max)
        v_min = tf.math.maximum(v_min_raw, 0)
        v_max = tf.math.minimum(v_max_raw, self.depth_image_size[1])

        return tf.stack([u_min, v_min, u_max, v_max], axis=-1)

    def compute_offsets(self, images, joints):
        """
            Computes offsets for each joint coordinate.
            All offsets are normalized to [-1, 1] if the given
            UVZ coords are normalized to [0, 1], and if the
            input images are also normalized to range [0, 1].

            !!! Offsets have to be computed from cropped resized
            normalized coordinates. !!!
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

        u_offsets = x[:, tf.newaxis, :, :]
        u_offsets = tf.tile(u_offsets, [1, self.image_out_size, 1, 1])

        v_offsets = y[:, :, tf.newaxis, :]
        v_offsets = tf.tile(v_offsets, [1, 1, self.image_out_size, 1])

        z_im = tf.image.resize(images, [self.image_out_size, self.image_out_size],
                               method=tf.image.ResizeMethod.BILINEAR)
        z_coords = joints[..., 2]
        z_coords = z_coords[:, tf.newaxis, tf.newaxis, :]
        z_offsets = z_coords - z_im
        return tf.stack([u_offsets, v_offsets, z_offsets], axis=-1)

    def offset_coord(self, joints_single_coord, n_joints):
        x = tf.linspace(0, 1, num=self.image_out_size)  # [0, 1]
        x = tf.cast(x, tf.float32)
        x = x[tf.newaxis, :, tf.newaxis]  # shape = [1, out_size, 1]
        x = tf.tile(x, [1, 1, n_joints])  # shape = [1, out_size, n_joints]
        x -= tf.expand_dims(joints_single_coord, 1)  # [0 - u, 1 - u], shape = [None, out_size, n_joints]
        x *= -1  # [u, u - 1]
        return x

    def crop_images(self, images, bboxes):
        """
        Parameters
        ----------
        images  tf.Tensor shape=[None, 480, 640]
        bboxes  tf.Tensor shape=[None, 4]
            The last dim contains [left, top, right, bottom].
        """

        def crop(elems):
            image, bbox = elems
            left, top, right, bottom = bbox
            cropped_image = image[top:bottom, left:right]
            return tf.RaggedTensor.from_tensor(cropped_image, ragged_rank=2)

        cropped = tf.map_fn(tf.autograph.experimental.do_not_convert(crop), elems=[images, bboxes],
                            fn_output_signature=tf.RaggedTensorSpec(shape=[None, None, 1], dtype=images.dtype))
        return cropped


def try_dataset_genereator():
    gen = DatasetGenerator(None, image_in_size=96, image_out_size=24, camera=Camera('bighand'))
    images = tf.ones([4, 480, 640])
    bboxes = tf.constant([[100, 120, 200, 230], [20, 50, 60, 80], [0, 0, 200, 200], [200, 200, 630, 470]])
    images, bboxes = gen.crop_images(images, bboxes)
    return images, bboxes


if __name__ == "__main__":
    # offsets = gen.compute_offsets(joints)
    try_dataset_genereator()
    pass