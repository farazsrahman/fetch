import numpy as np
from pydrake.all import cos, sin
from pydrake.math import RollPitchYaw, RotationMatrix, sin, cos
from pydrake.symbolic import if_then_else, Expression
from tamols import TAMOLSState
from helpers import (
    evaluate_spline_position, evaluate_spline_velocity, 
    evaluate_angular_momentum_derivative, evaluate_height_at_xy,
    evaluate_height_gradient, evaluate_smoothed_height_gradient
)

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

        for tau in tmls.taus_to_check:
            vel = evaluate_spline_velocity(tmls, a_k, tau, T_k)[0:3]

            for dim in range(3):
                total_cost += (vel[dim] - tmls.ref_vel[dim])**2

    tmls.prog.AddQuadraticCost(total_cost)

    
