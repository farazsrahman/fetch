import numpy as np
from pydrake.all import cos, sin


# SPLINE

def evaluate_spline_position(tmls, coeffs, tau):
    return np.sum([coeffs[:, i] * tau**i for i in range(tmls.spline_order)], axis=0)

def evaluate_spline_positions(tmls, coeffs, taus):
    taus_powers = np.array([taus**i for i in range(tmls.spline_order)])
    return np.sum(coeffs[:, :, np.newaxis] * taus_powers, axis=1).T

def evaluate_spline_velocity(tmls, coeffs, tau, T):
    # returns vel and euler rates (0:3 and 3:6 respectively)
    return np.sum([i * coeffs[:, i] * tau**(i - 1) / T for i in range(1, tmls.spline_order)], axis=0)
    
def evaluate_spline_acceleration(tmls, coeffs, tau, T):
    return np.sum([i * (i - 1) * coeffs[:, i] * tau**(i - 2) / (T * T) for i in range(2, tmls.spline_order)], axis=0)[:3]

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

def evaluate_height_at_xy(tmls, x, y):
    return 0.0

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
