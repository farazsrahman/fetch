import numpy as np
from pydrake.all import cos, sin
from pydrake.symbolic import if_then_else, Expression
from pydrake.math import RollPitchYaw


# SPLINE

def evaluate_spline_position(tmls, coeffs, tau):
    return np.sum([coeffs[:, i] * tau**i for i in range(tmls.spline_order)], axis=0)

def evaluate_spline_positions(tmls, coeffs, taus):
    taus_powers = np.array([taus**i for i in range(tmls.spline_order)])
    return np.sum(coeffs[:, :, np.newaxis] * taus_powers, axis=1).T

def evaluate_spline_velocity(tmls, coeffs, tau):
    # returns vel and euler rates (0:3 and 3:6 respectively)
    return np.sum([i * coeffs[:, i] * tau**(i - 1) for i in range(1, tmls.spline_order)], axis=0)
    
def evaluate_spline_acceleration(tmls, coeffs, tau):
    return np.sum([i * (i - 1) * coeffs[:, i] * tau**(i - 2) for i in range(2, tmls.spline_order)], axis=0)[:3]

def evaluate_angular_momentum_derivative(tmls, tau):
    return np.zeros(3) # TODO: Implement this


# GAIT

def get_num_contacts(tmls, phase):
    return sum(tmls.gait_pattern['contact_states'][phase])

def get_contact_pairs(tmls, stance_feet):
    return [(i, j) for i in stance_feet for j in stance_feet if i < j]

def get_stance_feet(tmls, phase):
    return [i for i, in_contact in enumerate(tmls.gait_pattern['contact_states'][phase]) if in_contact]


# HEIGHT MAP

def evaluate_height_at_symbolic_xy(tmls, height_map, x, y):
    m, n = height_map.shape
    grid_size = 0.04
    offset = 1.0
    i = (x + offset) / grid_size
    j = (y + offset) / grid_size
    total_height = 0

    for k in range(m):
        for l in range(n):
            partial_height = if_then_else(abs(i - k) < 1, 
            if_then_else(abs(j - l) < 1, height_map[k, l] * (1 - abs(i - k)) * (1 - abs(j - l)), 0), 0)
            total_height += partial_height

    return total_height

def evaluate_smoothed_height_gradient(tmls, x, y):
    if tmls.height_map_smoothed is None:
        return np.zeros(2)
    _, grad = tmls.height_map_smoothed([x, y])
    return np.array(grad)

def evaluate_height_gradient(tmls, x, y):
    if tmls.height_map is None:
        return np.zeros(2)
    _, grad = tmls.height_map([x, y])
    return np.array(grad)

# math

def determinant(a, b, c):
    return a.dot(np.cross(b, c))


def get_R_B(phi_B):
    """
    Calculate rotation matrix R_B from ZYX-Euler angles
    Args:
        phi_B: [psi, theta, phi] ZYX-Euler angles of the base
    Returns:
        3x3 rotation matrix R_B
    """
    # Create RollPitchYaw object from Euler angles
    # Note: RollPitchYaw expects [roll, pitch, yaw] = [phi, theta, psi]
    # So we need to reverse the order from [psi, theta, phi]
    rpy = RollPitchYaw([phi_B[2], phi_B[1], phi_B[0]])
    
    # Get rotation matrix
    R_B = rpy.ToRotationMatrix().matrix()
    return R_B
