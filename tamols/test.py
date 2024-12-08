import numpy as np
from pydrake.all import MathematicalProgram, Solve
from pydrake.solvers import SnoptSolver
from pydrake.all import Solve
from tamols import TAMOLSState, setup_variables
from constraints import (
    add_initial_constraints, 
    add_dynamics_constraints, 
    add_giac_constraints,
    add_friction_cone_constraints,
    add_kinematic_constraints,
)
from costs import (
    add_tracking_cost, 
    add_foot_collision_cost,
    add_test_cost,
    add_foothold_on_ground_cost,
    add_base_pose_alignment_cost,
    add_nominal_kinematic_cost,
    add_edge_avoidance_cost
)
from plotting_helpers import *
from map_processing import *
import manual_heightmaps as mhm

def setup_test_state(tmls: TAMOLSState):
     # Create a TAMOLSState instance

    tmls.base_pose = np.array([0, 0, 0.15, 0, 0, 0])  # Example initial base pose
    tmls.base_vel = np.array([0, 0, 0, 0, 0, 0])   # Example initial base velocity
    tmls.p_meas = np.array([
        [0.2, 0.1, 0],  # Front left leg
        [0.2, -0.1, 0], # Front right leg
        [-0.2, 0.1, 0], # Rear left leg
        [-0.2, -0.1, 0] # Rear right leg
    ])  # Reasonable initial foot positions

    elevation_map = mhm.get_platform_heightmap(tmls)
    # elevation_map = mhm.get_heightmap_with_holes()
    
    h_s1, h_s2, gradients = process_height_maps(elevation_map)

    tmls.h = elevation_map
    tmls.h_s1 = h_s1
    tmls.h_s2 = h_s2

    tmls.h_grad_x, tmls.h_grad_y = gradients['h']
    tmls.h_s1_grad_x, tmls.h_s1_grad_y = gradients['h_s1']
    tmls.h_s2_grad_x, tmls.h_s2_grad_y = gradients['h_s2']

    tmls.ref_vel = np.array([0.1, 0, 0])
    tmls.ref_angular_momentum = np.array([0, 0, 0])

  
    # single spline / phase
    tmls.gait_pattern = {
        'phase_timing': [0, 0.4, 0.8, 1.2, 1.6, 2.0],  # Adjusted phase timings
        'contact_states': [
            [1, 1, 1, 1],
            [1, 0, 1, 0],
            [1, 1, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 1],
        ],
        
        # boolean array of whether the foot is at the final position in the i-th phase
        # used to determine if p or p_meas should be used
        'at_des_position': [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 1],
        ],
    }
    

if __name__ == "__main__":

    # SETUP
    tmls = TAMOLSState()
    setup_test_state(tmls)
    setup_variables(tmls)

    # test specifc - hard coding final foot holds
    for leg_idx, pos in enumerate([[0.4, 0.1, 0], [0.4, -0.1, 0], [0, 0.1, 0], [0, -0.1, 0]]):
        for dim in range(2): # just x, y pos (z handled by foot on ground cost
            c = tmls.prog.AddLinearConstraint(tmls.p[leg_idx, dim] == pos[dim])
            tmls.test_constraints.append(c)

    # CONSTRAINTS
    add_initial_constraints(tmls)
    # add_dynamics_constraints(tmls)
    add_kinematic_constraints(tmls) # for some reason problem becomes infeasible without this
    add_giac_constraints(tmls)
    add_friction_cone_constraints(tmls)

    
    # COSTS
    # add_tracking_cost(tmls)
    add_foothold_on_ground_cost(tmls)
    # add_nominal_kinematic_cost(tmls)
    # add_base_pose_alignment_cost(tmls)
    # add_edge_avoidance_cost(tmls)


    # add_foot_collision_cost(tmls)
    # add_test_cost(tmls)

    solver = SnoptSolver()

    print("Starting solve")
    tmls.result = solver.Solve(tmls.prog)
    # result = Solve(tmls.prog)
    
    # Check if the problem is feasible
    if tmls.result.is_success():
        print("Optimization problem is feasible.")

        tmls.optimal_footsteps = tmls.result.GetSolution(tmls.p)
        num_phases = len(tmls.gait_pattern['phase_timing']) - 1
        tmls.optimal_spline_coeffs = [tmls.result.GetSolution(tmls.spline_coeffs[i]) for i in range(num_phases)]

        plot_optimal_solutions_interactive(tmls)
        save_optimal_solutions(tmls)

    else:
        print("Optimization problem is not feasible.")
        print("Solver result code:", tmls.result.GetInfeasibleConstraints(tmls.prog))

   