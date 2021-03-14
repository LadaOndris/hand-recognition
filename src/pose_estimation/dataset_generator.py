import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from src.utils.camera import Camera
from src.utils.plots import plot_joints_2d
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
        self.image_in_size = [image_in_size, image_in_size]
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

        resized_uv = normalized_uv * self.image_in_size
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
            images, ignore_bboxes, self.xyz_global = self.iterator.get_next()
        else:
            images, self.xyz_global = self.iterator.get_next()
        uv_global = self.camera.world_to_pixel(self.xyz_global)[..., :2]

        if self.augment:
            images, uv_global = self.augment_batch(images, uv_global)
        self.bboxes = self.extract_bboxes(uv_global)
        self.bboxes = self.square_bboxes(self.bboxes)
        self.cropped_imgs, bcubes = self.refine_and_crop_3d(images, self.bboxes, cube_size=(180, 180, 180))
        self.bboxes = tf.concat([bcubes[..., :2], bcubes[..., 3:5]], axis=-1)

        uv_cropped = uv_global - tf.cast(self.bboxes[:, tf.newaxis, :2], dtype=tf.float32)
        z_cropped = self.xyz_global[..., 2:3] - bcubes[..., 2:3]
        resized_imgs, resized_uv, self.resize_coeffs = self.resize_images_and_coords(
            self.cropped_imgs, uv_cropped, self.bboxes, target_size=self.image_in_size)

        self.normalized_imgs, normalized_uvz = self.normalize(resized_imgs, resized_uv, self.xyz_global[..., 2:3])
        offsets = self.compute_offsets(self.normalized_imgs, normalized_uvz)
        return self.normalized_imgs, [normalized_uvz, offsets]

    def resize_images_and_coords(self, cropped_imgs, coords_uv, bboxes, target_size):
        def _resize(img):
            return tf.image.resize(img.to_tensor(), target_size)

        resized_imgs = tf.map_fn(_resize, cropped_imgs,
                                 fn_output_signature=tf.TensorSpec(shape=(target_size[0], target_size[1], 1),
                                                                   dtype=tf.float32))
        cropped_imgs_sizes_u = bboxes[..., 2] - bboxes[..., 0]  # right - left
        cropped_imgs_sizes_v = bboxes[..., 3] - bboxes[..., 1]  # bottom - top
        cropped_imgs_sizes = tf.stack([cropped_imgs_sizes_u, cropped_imgs_sizes_v], axis=-1)
        resize_coeffs = tf.cast(target_size / cropped_imgs_sizes, dtype=tf.float32)
        resized_uv = resize_coeffs[:, tf.newaxis, :] * coords_uv
        return resized_imgs, resized_uv, resize_coeffs

    def refine_and_crop_3d(self, full_image, bbox, refine_iters=3, cube_size=(250, 250, 250)):
        """
        Refines the bounding box of the detected hand
        by iteratively finding its center of mass and
        cropping it in all three axes.

        Parameters
        ----------
        full_image
        bbox
        refine_iters
        cube_size

        Returns
        -------
        Tuple having (cropped_images, bcubes)
            cropped_images is a tf.RaggedTensor(shape=[batch_size, None, None, 1])
            bcubes is a tf.Tensor(shape=[batch_size, 6])
        """
        cropped = self.crop_bbox(full_image, bbox)
        coms = self.compute_coms(cropped, offsets=bbox[..., :2])
        coms = self.refine_coms(full_image, coms, iters=refine_iters, cube_size=cube_size)

        # Get the cube in UVZ around the center of mass
        bcube = self.com_to_bcube(coms, size=cube_size)
        # Crop the area defined by bcube from the orig image
        cropped = self.crop_bcube(full_image, bcube)
        return cropped, bcube

    def compute_coms(self, images, offsets):
        """
        Calculates the center of mass of the given image.
        Does not take into account the actual values of the pixels,
        but rather treats the pixels as either background, or something.

        Parameters
        ----------
        images : tf.RaggedTensor of shape [batch_size, None, None, 1]
            A batch of images.

        Returns
        -------
        center_of_mass : tf.Tensor of shape [batch_size, 3]
            Represented in UVZ coordinates.
        """
        com_local = tf.map_fn(self.center_of_mass, images,
                              fn_output_signature=tf.TensorSpec(shape=[3], dtype=tf.float32))

        # Adjust the center of mass coordinates to orig image space (add U, V offsets)
        com_uv_global = com_local[..., :2] + tf.cast(offsets, tf.float32)
        com_z = com_local[..., 2:3]
        coms = tf.concat([com_uv_global, com_z], axis=-1)
        return coms

    def center_of_mass(self, image):
        """
        Calculates the center of mass of the given image.
        Does not take into account the actual values of the pixels,
        but rather treats the pixels as either background, or something.

        Parameters
        ----------
        image : tf.Tensor of shape [width, height, 1]

        Returns
        -------
        center_of_mass : tf.Tensor of shape [3]
            Represented in UVZ coordinates.
        """
        if type(image) is tf.RaggedTensor:
            image = image.to_tensor()
        # Create all coordinate pairs
        im_width, im_height = tf.shape(image)[:2]
        x = tf.range(im_width)
        y = tf.range(im_height)
        xx, yy = tf.meshgrid(x, y, indexing='ij')
        xx = tf.reshape(xx, [-1])
        yy = tf.reshape(yy, [-1])
        # Stack along a new axis to create pairs in the last dimension
        coords = tf.stack([xx, yy], axis=-1)
        coords = tf.cast(coords, tf.float32)  # [im_width * im_height, 2]

        image_mask = tf.cast(image > 0, dtype=tf.float32)
        image_mask_flat = tf.reshape(image_mask, [im_width * im_height, 1])
        # The total mass of the depth
        total_mass = tf.reduce_sum(image)
        nonzero_pixels = tf.math.count_nonzero(image_mask, dtype=tf.float32)
        # Multiply the coords with volumes and reduce to get UV coords
        volumes_vu = tf.reduce_sum(image_mask_flat * coords, axis=0)
        volumes_uvz = tf.stack([volumes_vu[1], volumes_vu[0], total_mass], axis=0)
        com_uvz = tf.math.divide_no_nan(volumes_uvz, nonzero_pixels)
        return com_uvz

    def refine_coms(self, full_image, com, iters, cube_size):
        for i in range(iters):
            # Get the cube in UVZ around the center of mass
            bcube = self.com_to_bcube(com, size=cube_size)
            # fig, ax = plt.subplots()
            # ax.imshow(full_image[0])
            # ax.scatter(com[0, 0], com[0, 1])
            # r = patches.Rectangle(bcube[0, :2], bcube[0, 3] - bcube[0, 0], bcube[0, 4] - bcube[0, 1], facecolor='none',
            #                       edgecolor='r', linewidth=2)
            # ax.add_patch(r)
            # plt.show()
            # Crop the area defined by bcube from the orig image
            cropped = self.crop_bcube(full_image, bcube)
            # Compute center of mass again from the new cropped image
            com = self.compute_coms(cropped, offsets=bcube[..., :2])
        return com

    def com_to_bcube(self, com, size):
        """
        For the given center of mass (UVZ),
        computes a bounding cube in UVZ coordinates.

        Projects COM to the world coordinates,
        adds size offsets and projects back to image coordinates.
        """
        com_xyz = self.camera.pixel_to_world(com)
        half_size = tf.constant(size, dtype=tf.float32) / 2
        # Do not subtact Z coordinate yet
        # The Z coordinates must stay the same for both points
        # in order for the projection to image plane to be correct
        half_size = tf.stack([half_size[0], half_size[1], 0], axis=0)
        bcube_start_xyz = com_xyz - half_size
        bcube_end_xyz = com_xyz + half_size
        bcube_start_uv = self.camera.world_to_pixel(bcube_start_xyz)[..., :2]
        bcube_end_uv = self.camera.world_to_pixel(bcube_end_xyz)[..., :2]
        bcube_start_z = com[..., 2:3] - half_size[2]
        bcube_end_z = com[..., 2:3] + half_size[2]
        bcube = tf.concat([bcube_start_uv, bcube_start_z, bcube_end_uv, bcube_end_z], axis=-1)
        return tf.cast(bcube, dtype=tf.int32)

    def crop_bcube(self, images, bcubes):
        """
        Crops the image using a bounding cube. It is
        similar to cropping with a bounding box, but
        a bounding cube also defines the crop in Z axis.

        Parameters
        ----------
        image  Image to crop from.
        bcube  Bounding cube in UVZ coordinates.

        Returns
        -------
            Cropped image as defined by the bounding cube.
        """

        #
        # Look out for crop out of boundaries!
        #

        def crop(elems):
            image, bcube = elems
            x_start, y_start, z_start, x_end, y_end, z_end = bcube
            # Modify bcube because we its invalid to index with negatives.
            x_start_bound = tf.maximum(x_start, 0)
            y_start_bound = tf.maximum(y_start, 0)
            x_end_bound = tf.minimum(self.depth_image_size[0], x_end)
            y_end_bound = tf.minimum(self.depth_image_size[1], y_end)
            cropped_image = image[x_start_bound:x_end_bound, y_start_bound:y_end_bound]
            z_start = tf.cast(z_start, tf.float32)
            z_end = tf.cast(z_end, tf.float32)
            cropped_image = tf.where(cropped_image < z_start, z_start, cropped_image)
            cropped_image = tf.where(cropped_image > z_end, 0, cropped_image)

            # Pad the cropped image if we were out of bounds
            padded_image = tf.pad(cropped_image, [[x_start_bound - x_start, x_end - x_end_bound],
                                                  [y_start_bound - y_start, y_end - y_end_bound],
                                                  [0, 0]])
            return tf.RaggedTensor.from_tensor(padded_image, ragged_rank=2)

        cropped = tf.map_fn(tf.autograph.experimental.do_not_convert(crop), elems=[images, bcubes],
                            fn_output_signature=tf.RaggedTensorSpec(shape=[None, None, 1], dtype=images.dtype))

        # uv_start = bcube[..., :2]
        # uv_end = bcube[..., 3:5]
        # bbox = tf.concat([uv_start, uv_end], axis=-1)
        # cropped = self.crop_bbox(image, bbox)  # shape [None, None, 1]
        # z_start, z_end = bcube[..., 3], bcube[..., 5]
        # cropped = tf.where(cropped < z_start, z_start, cropped)
        # cropped = tf.where(cropped > z_end, z_end, 0)
        return cropped

    def normalize(self, images, joints_uv, joints_z):
        # self.max_depth = tf.math.reduce_max(joints_z, axis=[1, 2])
        normalized_uv = joints_uv / self.image_in_size
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

    def crop_bbox(self, images, bboxes):
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
    images, bboxes = gen.crop_bbox(images, bboxes)
    return images, bboxes


if __name__ == "__main__":
    # offsets = gen.compute_offsets(joints)
    try_dataset_genereator()
    pass
