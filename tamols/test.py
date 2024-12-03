import numpy as np
import time
from pydrake.all import MathematicalProgram, Solve
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
from helpers import evaluate_spline_position, evaluate_spline_positions

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def setup_test_state():
     # Create a TAMOLSState instance
    tmls = TAMOLSState()
    
    tmls.base_pose = np.array([0, 0, 0.2, 0, 0, 0])  # Example initial base pose
    tmls.base_vel = np.array([0, 0, 0, 0, 0, 0])   # Example initial base velocity
    tmls.p_meas = np.array([
        [0.2, 0.1, 0],  # Front left leg
        [0.2, -0.1, 0], # Front right leg
        [-0.2, 0.1, 0], # Rear left leg
        [-0.2, -0.1, 0] # Rear right leg
    ])  # Reasonable initial foot positions

    tmls.height_map = np.zeros((30, 30))
    tmls.height_map_smoothed = np.zeros((30, 30))
    
    tmls.gait_pattern = {
        'phase_timing': [0, 1, 2, 3],  # Example phase timings
        'contact_states': [
            [1, 1, 1, 1],  # All legs in contact in phase 0
            [1, 0, 1, 0],  # Alternating legs in contact in phase 1
            [1, 1, 1, 1],  # All legs in contact in phase 2
            [0, 1, 0, 1]   # Alternating legs in contact in phase 3
        ]
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

def plot_optimal_solutions(optimal_footsteps, optimal_spline_coeffs, num_phases, tmls):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b', 'y']
    for i in range(optimal_footsteps.shape[0]):
        ax.scatter(optimal_footsteps[i, 0], optimal_footsteps[i, 1], optimal_footsteps[i, 2], label=f'Footstep {i+1}', color=colors[i % len(colors)])

    for i in range(num_phases):
        T = tmls.phase_durations[i]
        tau_values = np.linspace(0, 1, 100)

        spline_points = evaluate_spline_positions(tmls, optimal_spline_coeffs[i], tau_values)
        ax.plot(spline_points[:, 0], spline_points[:, 1], spline_points[:, 2], label=f'Spline Phase {i+1}', color=colors[i % len(colors)])
        
        # Add coordinate labels for the end of each spline with exact coordinates
        ax.text(spline_points[-1, 0], spline_points[-1, 1], spline_points[-1, 2], 
                f'End {i+1} ({spline_points[-1, 0]:.2f}, {spline_points[-1, 1]:.2f}, {spline_points[-1, 2]:.2f})', 
                color=colors[i % len(colors)], fontsize=8)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Optimal Base Pose and Footsteps')
    ax.legend()
    
    plt.savefig('tamols/out/optimal_base_pose_and_footsteps.png')

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

        plot_optimal_solutions(optimal_footsteps, optimal_spline_coeffs, num_phases, tmls)

        save_optimal_solutions(optimal_footsteps, optimal_spline_coeffs, num_phases)

    else:
        print("Optimization problem is not feasible.")



