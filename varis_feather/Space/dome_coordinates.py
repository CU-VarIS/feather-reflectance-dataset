import numpy as np
from scipy.spatial.transform import Rotation


class DomeCoordinates:
    """
    Coordinate frames:
    * **envMapYUp**: cartesian coordinates with Y up, Z towards the closest wall, X away from the computer
    * **domeThetaPhi**: spherical coordinates where phi=0 is?
    * **envMapUV**: image coordinates for plotting on the saved environment map image
    * **faceCapture**: cartesian with Z away from the computer, X towards the wall?
    * **materialZUp**: cartesian coordinates with Z being the normal of the sample holder
    """

    # R_envMapYUp_to_domeYUp = Rotation.from_euler('y', -0.5*np.pi)
    R_envMapYUp_to_domeYUp = Rotation.identity()
    R_domeYUp_to_envMapYUp = R_envMapYUp_to_domeYUp.inv()

    R_domeYUp_to_sampleW0 = Rotation.from_euler("y", -0.5 * np.pi)
    R_sampleW0_to_domeYUp = R_domeYUp_to_sampleW0.inv()

    Wi_index_angles = np.pi * 0.5 - np.arccos(np.linspace(0, 1, 8))

    # @staticmethod
    # def make_R_domeEnvYUp_to_sample(
    #     arm_1_angle: float = 0.0, arm_2_angle: float = 0.0, arm_3_angle: float = 0.0
    # ) -> Rotation:
    #     """
    #     1, 2, 3 map to X, Y, Z arms in dome space.
    #     """
    #     #if arm_2_angle != 0.0 or arm_3_angle != 0.0:
    #     #    raise NotImplementedError("Arm 2 and 3 rotation not implemented")
    #
    #     # Rotations ordered biggest to smallest motor arms.
    #     Ry = Rotation.from_euler("y", np.pi + arm_2_angle).as_matrix()
    #     Rx = Rotation.from_euler("x", arm_1_angle).as_matrix()
    #     Rz = Rotation.from_euler("z", arm_3_angle).as_matrix()
    #
    #     # For anisotropic, rotate bigger arm as theta (X arm) THEN smaller (Z arm) about the adjusted position.
    #     # (just stick y in the middle for now... probably should pass rotation order as param)
    #     # TODO Ideally the order would be a parameter argument like '123' for x then y then z
    #     R = Rx @ Ry @ Rz
    #
    #     R = Rotation.from_matrix(R)
    #
    #     return R

    @staticmethod
    def t_domeThetaPhi_to_envMapUV(theta_phi: np.ndarray) -> np.ndarray:
        # Unpack
        theta = theta_phi[:, 0]
        phi = theta_phi[:, 1]

        # Rotate 180 deg around dome up axis
        phi_envMap = phi + np.pi

        uv = np.zeros_like(theta_phi)
        # Cast to 0...1 range:
        # np.remainder(..., 1) will give positive values for negative angles
        uv[:, 0] = np.remainder(phi_envMap / (2 * np.pi), 1)  # u
        uv[:, 1] = theta / np.pi  # v
        return uv

    @staticmethod
    def t_domeThetaPhi_to_domeYUp(theta_phi: np.ndarray) -> np.ndarray:
        # Unpack
        theta = theta_phi[:, 0]
        phi = theta_phi[:, 1]

        # Convert to cartesian
        x = np.sin(theta) * np.cos(phi)
        y = np.cos(theta)
        z = np.sin(theta) * np.sin(phi)

        return np.stack([x, y, z], axis=1)

    @staticmethod
    def t_envMapYUp_to_domeThetaPhi(envMapYUp: np.ndarray) -> np.ndarray:
        # Unpack
        x = envMapYUp[:, 0]
        up = envMapYUp[:, 1]
        z = envMapYUp[:, 2]

        # Convert to positive spherical
        theta = np.remainder(np.arccos(up), np.pi)
        phi = np.remainder(np.arctan2(z, x), 2 * np.pi)

        return np.stack([theta, phi], axis=1)

    @classmethod
    def t_envMapYUp_to_envMapUV(cls, envMapYUp: np.ndarray) -> np.ndarray:
        return cls.t_domeThetaPhi_to_envMapUV(
            cls.t_envMapYUp_to_domeThetaPhi(envMapYUp)
        )

    @classmethod
    def t_envMapYUp_to_domeYUp(cls, domeYUp: np.ndarray) -> np.ndarray:
        return cls.R_envMapYUp_to_domeYUp.apply(domeYUp)

    @classmethod
    def t_domeYUp_to_envMapYUp(cls, envMapYUp: np.ndarray) -> np.ndarray:
        return cls.R_domeYUp_to_envMapYUp.apply(envMapYUp)

    @classmethod
    def t_domeYUp_to_envMapUV(cls, domeYUp: np.ndarray) -> np.ndarray:
        return cls.t_domeThetaPhi_to_envMapUV(
            cls.t_envMapYUp_to_domeThetaPhi(cls.t_domeYUp_to_envMapYUp(domeYUp))
        )

    @classmethod
    def t_domeYUp_to_sampleW0(cls, domeYUp: np.ndarray) -> np.ndarray:
        return cls.R_domeYUp_to_sampleW0.apply(domeYUp)
