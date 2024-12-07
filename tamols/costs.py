import numpy as np
from tamols import TAMOLSState
from pydrake.symbolic import if_then_else, Expression
from pydrake.math import exp, abs, sqrt
from helpers import (
    evaluate_spline_position, evaluate_spline_velocity, 
    evaluate_angular_momentum_derivative, evaluate_height_at_symbolic_xy,
    evaluate_height_gradient, evaluate_smoothed_height_gradient, get_R_B
)
from pydrake.symbolic import floor, ExtractVariablesFromExpression, Expression


def add_tracking_cost(tmls: TAMOLSState):
    """Cost to track reference trajectory"""
    print("Adding tracking cost...")
    if tmls.ref_vel is None:
        raise ValueError("Reference velocity not set")

    num_phases = len(tmls.phase_durations)
    total_cost = 0

    for phase in range(num_phases):
        a_k = tmls.spline_coeffs[phase]
        T_k = tmls.phase_durations[phase]

        for tau in np.linspace(0, T_k, tmls.tau_sampling_rate+1)[:tmls.tau_sampling_rate]:
            vel = evaluate_spline_velocity(tmls, a_k, tau)[0:3]

            for dim in range(3):
                weight = 0.3
                total_cost += weight * (vel[dim] - tmls.ref_vel[dim])**2

    tmls.prog.AddQuadraticCost(total_cost)
   
def add_foot_collision_cost(tmls: TAMOLSState):
    print("Adding foot collision cost...")
    raise NotImplementedError("add_foot_collision_cost is not yet implemented")

#     num_legs = tmls.num_legs
#     min_foot_distance = tmls.min_foot_distance
#     total_cost = Expression(0)


#     # test with leg 1 and 2
#     distance = sqrt((tmls.p[0] - tmls.p[1]).dot(tmls.p[0] - tmls.p[1]) + 1e-6) 

#     penalty = (distance - min_foot_distance)**2
#     total_cost += penalty


#     # for i in range(num_legs):
#     #     for j in range(i + 1, num_legs):
#     #         distance = np.linalg.norm(tmls.p[i] - tmls.p[j])
#     #         penalty = if_then_else(distance > min_foot_distance, (distance - min_foot_distance)**2, 0)
#     #         total_cost += penalty

#     tmls.prog.AddCost(total_cost)

def add_test_cost(tmls: TAMOLSState):
    print("Adding test cost...")
    total_cost = 0
    num_legs = tmls.num_legs

    footstep_1 = tmls.p[0]
    target_point = np.array([0.5, 0.5, 0])
    distance = np.linalg.norm(footstep_1 - target_point)
    total_cost += distance**2 

    tmls.prog.AddCost(total_cost)


def add_foothold_on_ground_cost(tmls: TAMOLSState):
    """Cost to keep footholds on ground"""
    print("Adding foothold cost...")

    e_z = np.array([0., 0., 1.])
    total_cost = 0

    for i in range(tmls.num_legs):
        # Create continuous height approximation using bilinear interpolation
        h_pi = evaluate_height_at_symbolic_xy(tmls, tmls.h, tmls.p[i][0], tmls.p[i][1])
        
        # Add to cost using interpolated height
        cost = (h_pi - e_z.dot(tmls.p[i]))**2
        total_cost += cost

    tmls.prog.AddCost(total_cost)

def add_base_pose_alignment_cost(tmls: TAMOLSState):
    """Cost to align base on ground"""
    print("Adding base pose alignment cost...")

    e_z = np.array([0., 0., 1.])
    total_cost = 0

    l_des = np.array([0., 0., tmls.h_des])

    num_phases = len(tmls.phase_durations)
    for phase in range(num_phases):
        a_k = tmls.spline_coeffs[phase]
        T_k = tmls.phase_durations[phase]

        for tau in np.linspace(0, T_k, tmls.tau_sampling_rate+1)[:tmls.tau_sampling_rate]:
            p_B = evaluate_spline_position(tmls, a_k, tau)[:3]
            phi_B = evaluate_spline_position(tmls, a_k, tau)[3:6]

            R_B = get_R_B(phi_B)

            hs2_pB = evaluate_height_at_symbolic_xy(tmls, tmls.hs2, p_B[0], p_B[1])
            for i in range(tmls.num_legs):
                base_minus_leg = p_B + R_B.dot(tmls.hip_offsets[i]) - l_des
                cost = (e_z.dot(base_minus_leg) - hs2_pB)**2
                total_cost += cost

    tmls.prog.AddCost(total_cost)

