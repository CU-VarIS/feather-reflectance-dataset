import numpy as np

def u_to_theta(u):
    """Range: [0, pi/2]"""
    return 0.5 * np.pi * u * u


def u_to_phi(u):
    """Range: [-pi, pi]"""
    return np.pi * (2 * u - 1)


def spherical_to_sample_space(theta_phi):
    theta = theta_phi[..., 0]
    phi = theta_phi[..., 1]
    sin_theta = np.sin(theta)
    return np.stack( [sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)], axis=-1 )

