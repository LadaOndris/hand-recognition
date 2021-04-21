import numpy as np
import tensorflow as tf


class Camera:

    def __init__(self, cam_type: str):
        self.dataset = cam_type
        self.extrinsic_matrix = tf.eye(4)
        self.focal_length = None
        self.principal_point = None
        self.image_size = None
        self.intrinsic_matrix = None
        self.projection_matrix = None
        self.invr_projection_matrix = None
        self.depth_unit = None

        cam_type = cam_type.lower()
        if cam_type == 'bighand':
            self.set_bighand_sr300()
        elif cam_type == 'msra':
            self.set_msra_creative_interactive()
        elif cam_type == 'custom' or cam_type == 'sr305':
            self.set_sr305()
        elif cam_type == 'd415':
            self.set_d415()
        else:
            raise NotImplementedError('Unknown dataset')
        self.create_projection_matrices()

    def set_msra_creative_interactive(self):
        self.focal_length = [241.42, 241.42]
        self.principal_point = [160, 120]
        self.image_size = [320, 240]

    def set_bighand_sr300(self):
        self.focal_length = [475.065948, 475.065857]  # [476.0068, 476.0068]  # [588.235, 587.084]
        self.principal_point = [315.944855, 245.287079]
        self.image_size = [640, 480]
        self.extrinsic_matrix = tf.constant(
            [[0.999988496304, -0.00468848412856, 0.000982563360594, 0],
             [0.00469115935266, 0.999985218048, -0.00273845880292, 0],
             [-0.000969709653873, 0.00274303671904, 0.99999576807, 0],
             [0, 0, 0, 1]], dtype=tf.float32)
        self.depth_unit = 0.001

    def set_sr305(self):
        # self.focal_length = [588.235, 587.084]
        self.focal_length = [476.0068, 476.0068]
        self.principal_point = [313.6830139, 242.7547302]
        self.image_size = [640, 480]
        self.depth_unit = 0.00012498664727900177  # 0.125 mm

    def set_d415(self):
        self.focal_length = [592.138, 592.138]
        self.principal_point = [313.79, 238.076]
        self.image_size = [640, 480]
        self.depth_unit = 0.001  # 1 mm

    def create_projection_matrices(self):
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
        self.intrinsic_matrix = tf.constant([[self.focal_length[0], 0, self.principal_point[0], 0],
                                             [0, self.focal_length[1], self.principal_point[1], 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], dtype=tf.float32)
        # Compute WorldToPixel projection matrix
        self.projection_matrix = tf.matmul(self.intrinsic_matrix, self.extrinsic_matrix)
        self.invr_projection_matrix = tf.linalg.inv(self.projection_matrix)

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
        points_xyz = tf.cast(points_xyz, tf.float32)

        # Add ones for all points
        new_shape = [points_xyz.shape[0], points_xyz.shape[1], 1]
        points = tf.concat([points_xyz, tf.ones(new_shape, dtype=points_xyz.dtype)], axis=-1)
        # Project onto image plane
        projected_points = tf.matmul(self.projection_matrix, points, transpose_b=True)
        projected_points = tf.transpose(projected_points, [0, 2, 1])[..., :3]

        # Devide by Z
        uv = projected_points[..., :2] / projected_points[..., 2:3]
        uvz = tf.concat([uv, projected_points[..., 2:3]], axis=-1)

        if tf.rank(coords_xyz) == 2:
            uvz = tf.squeeze(uvz, axis=0)
        return uvz

    def pixel_to_world(self, coords_uvz):
        if tf.rank(coords_uvz) == 2:
            points_uvz = coords_uvz[tf.newaxis, ...]
        else:
            points_uvz = coords_uvz
        points_uvz = tf.cast(points_uvz, tf.float32)

        multiplied_uv = points_uvz[..., 0:2] * points_uvz[..., 2:3]
        ones = tf.ones([points_uvz.shape[0], points_uvz.shape[1], 1], dtype=points_uvz.dtype)
        multiplied_uvz1 = tf.concat([multiplied_uv, points_uvz[..., 2:3], ones], axis=-1)
        tranposed_xyz = tf.matmul(self.invr_projection_matrix, multiplied_uvz1, transpose_b=True)
        xyz = tf.transpose(tranposed_xyz, [0, 2, 1])[..., :3]

        if tf.rank(coords_uvz) == 2:
            xyz = tf.squeeze(xyz, axis=0)
        return xyz
