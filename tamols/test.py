import numpy as np
from pydrake.all import MathematicalProgram, Solve
from pydrake.solvers import SnoptSolver
from pydrake.all import Solve
from tamols import TAMOLSState, setup_variables
from constraints import (
    add_initial_constraints, 
    add_dynamics_constraints, 
    add_kinematic_constraints,
    add_giac_constraints,
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

    # elevation_map = mhm.get_platform_heightmap()
    elevation_map = mhm.get_heightmap_with_holes()
    
    h_s1, h_s2, gradients = process_height_maps(elevation_map)

    tmls.h = elevation_map
    tmls.h_s1 = h_s1
    tmls.h_s2 = h_s2

    tmls.h_grad_x, tmls.h_grad_y = gradients['h']
    tmls.h_s1_grad_x, tmls.h_s1_grad_y = gradients['h_s1']
    tmls.h_s2_grad_x, tmls.h_s2_grad_y = gradients['h_s2']

    tmls.ref_vel = np.array([0.10, 0, 0])
    tmls.ref_angular_momentum = np.array([0, 0, 0])

  
    # single spline / phase
    tmls.gait_pattern = {
        'phase_timing': [0, 0.05, 0.50, 0.55, 1.00, 1.05],  # Adjusted phase timings
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
    # add_initial_constraints(tmls)
    # add_dynamics_constraints(tmls)
    # add_kinematic_constraints(tmls) # for some reason problem becomes infeasible without this
    # add_giac_constraints(tmls)

    
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

    # Create meshgrid for 3D plotting
    x = np.arange(0, 24)
    y = np.arange(0, 24)
    X, Y = np.meshgrid(x, y)

    # Create 3D visualization
    fig = plt.figure(figsize=(20, 6))

    # Original height map
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, tmls.h, cmap='terrain', edgecolor='none')
    ax1.set_title('TMLS Height Map (h)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    ax1.set_xlabel('Grid X')
    ax1.set_ylabel('Grid Y')
    ax1.set_zlabel('Height (m)')

    # h_s1 map
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, Y, tmls.h_s1, cmap='terrain', edgecolor='none')
    ax2.set_title('TMLS Gaussian Filtered (h_s1)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    ax2.set_xlabel('Grid X')
    ax2.set_ylabel('Grid Y')
    ax2.set_zlabel('Height (m)')

    # h_s2 map
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X, Y, tmls.h_s2, cmap='terrain', edgecolor='none')
    ax3.set_title('TMLS Virtual Floor (h_s2)')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
    ax3.set_xlabel('Grid X')
    ax3.set_ylabel('Grid Y')
    ax3.set_zlabel('Height (m)')

    # Adjust the view angle for better visualization
    for ax in [ax1, ax2, ax3]:
        ax.view_init(elev=30, azim=45)
        ax.set_box_aspect([1,1,0.5])

    plt.tight_layout()

    # Save the figure with high DPI
    plt.savefig('tmls_height_maps.png', dpi=300, bbox_inches='tight')
    plt.close()




