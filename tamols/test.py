import numpy as np
import time
from pydrake.all import MathematicalProgram, Solve
from tamols import TAMOLSState, setup_variables
from constraints import add_initial_constraints, add_dynamics_constraints, add_kinematic_constraints, add_gait_constraints
from costs import add_costs
# from helpers import set_height_maps

def test_optimization():
    # Create a TAMOLSState instance
    tmls = TAMOLSState()
    
    # Set initial state
    tmls.base_pose = np.array([0, 0, 0, 0, 0, 0])  # Example initial base pose
    tmls.base_vel = np.array([0, 0, 0, 0, 0, 0])   # Example initial base velocity
    tmls.p_meas = np.zeros((tmls.num_legs, 3))     # Example initial foot positions
    
    # Set gait pattern
    tmls.gait_pattern = {
        'phase_timing': [0, 1, 2, 3],  # Example phase timings
        'contact_states': [
            [1, 1, 1, 1],  # All legs in contact in phase 0
            [1, 0, 1, 0],  # Alternating legs in contact in phase 1
            [1, 1, 1, 1],  # All legs in contact in phase 2
            [0, 1, 0, 1]   # Alternating legs in contact in phase 3
        ]
    }
    
    # Run setup to initialize variables

    # TIMER START
    start_time = time.time()
    setup_variables(tmls)
    print(f"Setup variables took {time.time() - start_time:.4f} seconds.")
    
    # Set height maps
    tmls.height_map = np.zeros((30, 30))
    tmls.height_map_smoothed = np.zeros((30, 30))
    
    # Add constraints
    # TIMER LOG
    start_time = time.time()
    add_initial_constraints(tmls)
    print(f"Add initial constraints took {time.time() - start_time:.4f} seconds.")
    
    add_dynamics_constraints(tmls)
    print(f"Add dynamics constraints took {time.time() - start_time:.4f} seconds.")
    
    add_kinematic_constraints(tmls)
    print(f"Add kinematic constraints took {time.time() - start_time:.4f} seconds.")
    
    add_gait_constraints(tmls)
    print(f"Add gait constraints took {time.time() - start_time:.4f} seconds.")

    # Add costs
    # TIMER LOG
    start_time = time.time()
    add_costs(tmls)
    print(f"Add costs took {time.time() - start_time:.4f} seconds.")
    
    # Solve the optimization problem
    # TIMER LOG
    start_time = time.time()
    result = Solve(tmls.prog)
    print(f"Solve optimization problem took {time.time() - start_time:.4f} seconds.")
    
    # Check if the problem is feasible
    if result.is_success():
        print("Optimization problem is feasible.")
    else:
        print("Optimization problem is not feasible.")

if __name__ == "__main__":
    test_optimization()
