import numpy as np
from pydrake.all import cos, sin


# spline evaluation

def evaluate_spline_position(tmls, coeffs, tau):
    return np.sum([coeffs[:, i] * tau**i for i in range(tmls.spline_order)], axis=0)

def evaluate_spline_positions(tmls, coeffs, taus):
    taus_powers = np.array([taus**i for i in range(tmls.spline_order)])
    return np.sum(coeffs[:, :, np.newaxis] * taus_powers, axis=1).T

def evaluate_spline_velocity(tmls, coeffs, tau, T):
    return np.sum([i * coeffs[:, i] * tau**(i - 1) / T for i in range(1, tmls.spline_order)], axis=0)
    
def evaluate_spline_acceleration(tmls, coeffs, tau, T):
    return np.sum([i * (i - 1) * coeffs[:, i] * tau**(i - 2) / (T * T) for i in range(2, tmls.spline_order)], axis=0)[:3]

def evaluate_angular_momentum_derivative(tmls, tau):
    return np.zeros(3) # TODO: Implement this


# contacts and gait pattern

def get_num_contacts(tmls, phase):
    return sum(tmls.gait_pattern['contact_states'][phase])

def get_contact_pairs(tmls, stance_feet):
    return [(i, j) for i in stance_feet for j in stance_feet if i < j]

def get_stance_feet(tmls, phase):
    return [i for i, in_contact in enumerate(tmls.gait_pattern['contact_states'][phase]) if in_contact]


# height map

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







# # GENERAL HELPERS

# def get_phase_at_time(tmls, t, gait_pattern):
#     """Get which phase index and normalized time within phase"""
#     phase_times = gait_pattern['phase_timing']
#     for i in range(len(phase_times) - 1):
#         if phase_times[i] <= t <= phase_times[i + 1]:
#             tau = (t - phase_times[i]) / (phase_times[i + 1] - phase_times[i])
#             return i, tau
#     raise ValueError(f"Time {t} not within gait pattern")

# def evaluate_spline(tmls, t, gait_pattern):
#     """Evaluate spline at time t"""
#     phase_idx, tau = get_phase_at_time(tmls, t, gait_pattern)
#     coeffs = tmls.vars['spline_coeffs'][phase_idx]
    
#     pos = np.zeros(tmls.base_dims)
#     vel = np.zeros(tmls.base_dims)
#     acc = np.zeros(tmls.base_dims)
    
#     for i in range(tmls.spline_order):
#         pos += coeffs[:, i] * tau**i
#         if i > 0:
#             vel += i * coeffs[:, i] * tau**(i - 1)
#         if i > 1:
#             acc += i * (i - 1) * coeffs[:, i] * tau**(i - 2)

# def evaluate_spline_velocity(tmls, coeffs, tau, T):
#     """Evaluate spline velocity at normalized time tau"""
#     vel = np.sum([i * coeffs[:, i] * tau**(i - 1) / T for i in range(1, tmls.spline_order)], axis=0)
#     return vel

# def set_height_maps(tmls, height_map, height_map_smoothed, x_min, x_max, y_min, y_max):
#     """Set the height maps and create interpolators"""
#     tmls.height_map = height_map
#     tmls.height_map_smoothed = height_map_smoothed
#     tmls.x_range = [x_min, x_max]
#     tmls.y_range = [y_min, y_max]

# def evaluate_smoothed_height_at_xy(tmls, x, y):
#     """Evaluate smoothed terrain height at given x,y point"""
#     if tmls.height_map_smoothed is None:
#         return 0.0
#     return float(tmls.height_map_smoothed([x, y]))

# def compute_rotation_matrix(rpy):
#     """Compute rotation matrix from roll-pitch-yaw angles"""
#     cr, cp, cy = np.cos(rpy)
#     sr, sp, sy = np.sin(rpy)
    
#     return np.array([
#         [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
#         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
#         [-sp, cp * sr, cp * cr]
#     ])
