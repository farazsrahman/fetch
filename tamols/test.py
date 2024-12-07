import numpy as np
from pydrake.all import MathematicalProgram, Solve
from pydrake.solvers import SnoptSolver
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
)
from plotting_helpers import *

def setup_test_state(tmls: TAMOLSState):
     # Create a TAMOLSState instance

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

    tmls.ref_vel = np.array([0.10, 0, 0])
    tmls.ref_angular_momentum = np.array([0, 0, 0])

  
    # single spline / phase
    tmls.gait_pattern = {
        'phase_timing': [0, 1.0, 2.0, 3.0, 4.0],  # Adjusted phase timings
        'contact_states': [
            [1, 0, 1, 0],
            [1, 1, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 1],
        ],
        
        # boolean array of whether the foot is at the final position in the i-th phase
        # used to determine if p or p_meas should be used
        'at_des_position': [
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
        for dim in [2]: # only enforcing z-position
            tmls.prog.AddLinearConstraint(tmls.p[leg_idx, dim] == pos[dim])

    # CONSTRAINTS
    add_initial_constraints(tmls)
    add_dynamics_constraints(tmls)
    add_kinematic_constraints(tmls) # for some reason problem becomes infeasible without this

    add_giac_constraints(tmls)

    
    # COSTS
    add_tracking_cost(tmls)
    # add_foot_collision_cost(tmls)
    # add_test_cost(tmls)


    # SOLVE
    print("\nSolving...")
    solver = SnoptSolver() # WHY THIS NOT FIND OPTIAL
    tmls.result = solver.Solve(tmls.prog)  
    # tmls.result = Solve(tmls.prog)
    
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
        print(tmls.result.GetInfeasibleConstraints(tmls.prog))



