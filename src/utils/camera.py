import numpy as np
import tensorflow as tf


class Camera:

    def __init__(self, dataset: str):
        self.dataset = dataset
        if dataset == 'bighand':
            self.set_intel_realsense_sr300()
        elif dataset == 'msra':
            self.set_intel_creative_interactive()
        elif dataset == 'custom':
            self.set_custom()
        else:
            raise NotImplementedError('Unknown dataset')

    def set_intel_creative_interactive(self):
        self.focal_length = [241.42, 241.42]
        self.principal_point = [160, 120]
        self.image_size = [320, 240]
        self.create_matrices()

    def set_intel_realsense_sr300(self):
        self.focal_length = [475.065948, 475.065857]  # [476.0068, 476.0068]  # [588.235, 587.084]
        self.principal_point = [315.944855, 245.287079]
        self.image_size = [640, 480]
        self.create_matrices()

    def set_custom(self):
        # self.focal_length = [588.235, 587.084]
        self.focal_length = [476.0068, 476.0068]
        self.principal_point = [313.6830139, 242.7547302]
        self.image_size = [640, 480]
        self.create_matrices()

    def create_matrices(self):
        # More information on intrinsic and extrinsic parameters
        # can be found at: https://en.wikipedia.org/wiki/Camera_resectioning#Intrinsic_parameters
        """
        self.extrinsic_matrix = tf.constant(
            [[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
             [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
             [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
             [0, 0, 0, 1]], dtype=tf.float32)

        self.extrinsic_matrix = tf.constant(
            [[1, 0, 0, 25.7],
             [0, 1, 0, 1.22],
             [0, 0, 1, 3.902],
             [0, 0, 0, 1]], dtype=tf.float32)
        """
        self.extrinsic_matrix = tf.constant(
            [[0.999988496304, -0.00468848412856, 0.000982563360594, 0],
             [0.00469115935266, 0.999985218048, -0.00273845880292, 0],
             [-0.000969709653873, 0.00274303671904, 0.99999576807, 0],
             [0, 0, 0, 1]], dtype=tf.float32)
        self.intrinsic_matrix = tf.constant([[self.focal_length[0], 0, self.principal_point[0]],
                                             [0, self.focal_length[1], self.principal_point[1]],
                                             [0, 0, 1]], dtype=tf.float32)
        self.invr_intrinsic_matrix = tf.linalg.inv(self.intrinsic_matrix)

    def world_to_pixel(self, coords_xyz):
        """
        Projects the given points through pinhole camera on a plane at a distance of focal_length.
        Returns
        -------
        tf.Tensor
            UVZ coordinates
        """
        if tf.rank(coords_xyz) == 2:
            points_xyz = coords_xyz[tf.newaxis, ...]
        else:
            points_xyz = coords_xyz

        if self.dataset == 'bighand':
            new_shape = [points_xyz.shape[0], points_xyz.shape[1], 1]
            points = tf.concat([points_xyz, tf.ones(new_shape, dtype=points_xyz.dtype)], axis=-1)
            extr_points = tf.matmul(self.extrinsic_matrix, points, transpose_b=True)
            points_xyz = tf.transpose(extr_points, [0, 2, 1])[..., :3]

        intr_points = tf.matmul(self.intrinsic_matrix, points_xyz, transpose_b=True)
        intr_points = tf.transpose(intr_points, [0, 2, 1])
        proj_points = intr_points[..., :2] / intr_points[..., 2:3]
        uvz = tf.concat([proj_points, intr_points[..., 2:3]], axis=-1)
        if tf.rank(coords_xyz) == 2:
            uvz = tf.squeeze(uvz, axis=0)
        return uvz

    def pixel_to_world(self, coords_uvz):
        if tf.rank(coords_uvz) == 2:
            points_uvz = coords_uvz[tf.newaxis, ...]
        else:
            points_uvz = coords_uvz

        multiplied_uv = points_uvz[..., 0:2] * points_uvz[..., 2:3]
        multiplied_uvz = tf.concat([multiplied_uv, points_uvz[..., 2:3]], axis=-1)
        tranposed_xyz = tf.matmul(self.invr_intrinsic_matrix, multiplied_uvz, transpose_b=True)
        xyz = tf.transpose(tranposed_xyz, [0, 2, 1])
        if tf.rank(coords_uvz) == 2:
            xyz = tf.squeeze(xyz, axis=0)
        return xyz
