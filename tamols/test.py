import numpy as np
import time
from pydrake.solvers import SnoptSolver
from tamols import TAMOLSState, setup_variables
from constraints import (
    add_initial_constraints, 
    add_dynamics_constraints, 
    add_kinematic_constraints, 
    add_gait_constraints
)
from costs import (
    add_robustness_cost, 
    add_foothold_ground_cost, 
    add_leg_collision_cost, 
    add_nominal_kinematics_cost, 
    add_base_alignment_cost, 
    add_edge_avoidance_cost, 
    add_previous_solution_cost, 
    add_tracking_cost, 
    add_smoothness_cost
)

from plotting_helpers import *



def setup_test_state():
     # Create a TAMOLSState instance
    tmls = TAMOLSState()
    
    tmls.base_pose = np.array([0, 0, 0.3, 0, 0, 0])  # Example initial base pose
    tmls.base_vel = np.array([0, 0, 0, 0, 0, 0])   # Example initial base velocity
    # tmls.p_meas = np.array([
    #     [0.2, 0.1, 0],  # Front left leg
    #     [0.2, -0.1, 0], # Front right leg
    #     [-0.2, 0.1, 0], # Rear left leg
    #     [-0.2, -0.1, 0] # Rear right leg
    # ])  # Reasonable initial foot positions

    # pathalogical
     # - without dynamics accurately picking whether to look at p_meas or p this does not matter 
    tmls.p_meas = np.array([ 
        [0.2, 0.1, .05],  # Front left leg
        [0.2, -0.1, 0], # Front right leg
        [-0.2, 0.1, 0], # Rear left leg
        [-0.2, -0.1, .05] # Rear right leg
    ])  # mid movement foot positions

    tmls.height_map = np.zeros((30, 30))
    tmls.height_map_smoothed = np.zeros((30, 30))
    

    tmls.gait_pattern = {
        'phase_timing': [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],  # Adjusted phase timings
        'contact_states': [
            [1, 1, 1, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 1]
        ],
        
        # boolean array of whether the foot is at the final position in the i-th phase
        # used to determine if p or p_meas should be used
        'at_des_position': [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 1]
        ],
    }

    return tmls

def save_optimal_solutions(optimal_footsteps, optimal_spline_coeffs, num_phases, filepath='tamols/out/optimal_solution.txt'):
        with open(filepath, 'w') as f:
            f.write("Optimal Footsteps:\n")
            for i in range(optimal_footsteps.shape[0]):
                f.write(f"Footstep {i+1}: {optimal_footsteps[i, 0]}, {optimal_footsteps[i, 1]}, {optimal_footsteps[i, 2]}\n")
            
            f.write("\nOptimal Spline Coefficients:\n")
            for i in range(num_phases):
                f.write(f"Spline Phase {i+1} Coefficients:\n")
                np.savetxt(f, optimal_spline_coeffs[i], fmt='%.6f')
                f.write("\n")



if __name__ == "__main__":
   
    tmls = setup_test_state()

    setup_variables(tmls)
    
    add_initial_constraints(tmls)

    add_dynamics_constraints(tmls)
    
    add_kinematic_constraints(tmls)

    add_gait_constraints(tmls)



    solver = SnoptSolver()
    result = solver.Solve(tmls.prog)
    
    
    # Check if the problem is feasible
    if result.is_success():
        print("Optimization problem is feasible.")
                
        # Extract optimal footstep positions
        optimal_footsteps = result.GetSolution(tmls.p); print(f"Optimal footsteps type: {type(optimal_footsteps)}, shape: {optimal_footsteps.shape}")
        
        num_phases = len(tmls.gait_pattern['phase_timing']) - 1
        optimal_spline_coeffs = [result.GetSolution(tmls.spline_coeffs[i]) for i in range(num_phases)]

        plot_optimal_solutions_interactive(optimal_footsteps, optimal_spline_coeffs, num_phases, tmls)

        save_optimal_solutions(optimal_footsteps, optimal_spline_coeffs, num_phases)

    else:
        print("Optimization problem is not feasible.")



