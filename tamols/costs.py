import numpy as np
from tamols import TAMOLSState
from pydrake.symbolic import if_then_else, Expression
from pydrake.math import exp, abs, sqrt
from helpers import (
    evaluate_spline_position, evaluate_spline_velocity, 
    evaluate_angular_momentum_derivative, evaluate_height_at_xy,
    evaluate_height_gradient, evaluate_smoothed_height_gradient
)


def add_tracking_cost(tmls: TAMOLSState):
    print("Adding tracking cost...")
    if tmls.ref_vel is None:
        raise ValueError("Reference velocity not set")

    num_phases = len(tmls.phase_durations)
    total_cost = 0

    for phase in range(num_phases):
        a_k = tmls.spline_coeffs[phase]
        T_k = tmls.phase_durations[phase]

        for tau in tmls.taus_to_check:
            vel = evaluate_spline_velocity(tmls, a_k, tau, T_k)[0:3]

            for dim in range(3):
                total_cost += (vel[dim] - tmls.ref_vel[dim])**2

    tmls.prog.AddQuadraticCost(total_cost)
   
def add_foot_collision_cost(tmls: TAMOLSState):
    print("Adding foot collision cost...")

    num_legs = tmls.num_legs
    min_foot_distance = tmls.min_foot_distance
    total_cost = Expression(0)


    # test with leg 1 and 2
    distance = sqrt((tmls.p[0] - tmls.p[1]).dot(tmls.p[0] - tmls.p[1]) + 1e-6) 

    penalty = (distance - min_foot_distance)**2
    total_cost += penalty


    # for i in range(num_legs):
    #     for j in range(i + 1, num_legs):
    #         distance = np.linalg.norm(tmls.p[i] - tmls.p[j])
    #         penalty = if_then_else(distance > min_foot_distance, (distance - min_foot_distance)**2, 0)
    #         total_cost += penalty

    tmls.prog.AddCost(total_cost)

def add_test_cost(tmls: TAMOLSState):
    print("Adding test cost...")
    total_cost = 0
    num_legs = tmls.num_legs

    footstep_1 = tmls.p[0]
    target_point = np.array([0.5, 0.5, 0])
    distance = np.linalg.norm(footstep_1 - target_point)
    total_cost += distance**2 

    tmls.prog.AddCost(total_cost)




