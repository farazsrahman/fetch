from dataclasses import dataclass, field
import numpy as np
from pydrake.all import (
    MathematicalProgram
)
from pydrake.math import RollPitchYaw, RotationMatrix
from typing import List, Dict, Tuple, Union


# STATE

@dataclass
class TAMOLSState:
    # Time and horizon parameters
    dt: float = 0.01
    horizon_steps: int = 100

    # Height map data
    height_map: np.ndarray = None
    height_map_smoothed: np.ndarray = None

    # Robot configuration
    num_legs: int = 4
    spline_order: int = 5
    base_dims: int = 6

    # Physical parameters
    mass: float = 1.0
    mu: float = 0.7
    inertia: np.ndarray = field(default_factory=lambda: np.diag([0.07, 0.26, 0.242]))

    # Leg configuration
    hip_offsets: np.ndarray = field(default_factory=lambda: np.array([
        [0.2, 0.15, 0],
        [0.2, -0.15, 0],
        [-0.2, 0.15, 0],
        [-0.2, -0.15, 0],
    ]))
    l_min: float = 0.08
    l_max: float = 0.25

    # Current state
    p_meas: np.ndarray = None
    base_pose: np.ndarray = None
    base_vel: np.ndarray = None

    # Optimization hyper-parameters TODO
    taus_to_check: List[float] = field(default_factory=lambda: [0.4, 0.9])
    weights: Dict[str, float] = field(default_factory=lambda: {
        'robustness_margin': 1.0,
        'footholds_on_ground': 1.0,
        'leg_collision_avoidance': 1.0,
        'nominal_kinematics': 1.0,
        'base_alignment': 1.0,
        'edge_avoidance': 1.0,
        'previous_solution_tracking': 1.0,
        'reference_trajectory_tracking': 1.0,
        'smoothness': 1.0,
    })

    # Optimization variables
    prog: MathematicalProgram = None
    spline_coeffs: List[np.ndarray] = None
    p: np.ndarray = None
    epsilon: np.ndarray = None

    # Desired configuration
    nominal_height: float = 0.4
    foot_radius: float = 0.02
    min_foot_distance: float = 0.1
    desired_height: float = 0.4

    reference_trajectory: np.ndarray = None
    gait_pattern: Dict[str, List[Union[float, List[int]]]] = None
    
    # Internal variables
    phase_durations: List[float] = None
    

# SETUP

def setup_variables(tmls: TAMOLSState):
    """Setup optimization variables based on gait phases"""
    tmls.prog = MathematicalProgram()
    
    phase_times = tmls.gait_pattern['phase_timing']
    num_phases = len(phase_times) - 1  # Number of intervals between timestamps
    tmls.phase_durations = [
        phase_times[i+1] - phase_times[i] 
        for i in range(num_phases)
    ]

    # Spline coefficients
    tmls.spline_coeffs = []
    for i in range(num_phases):
        coeffs = tmls.prog.NewContinuousVariables(
            tmls.base_dims, tmls.spline_order, 
            f'a_{i}'
        )
        tmls.spline_coeffs.append(coeffs)

        # tmls.prog.SetInitialGuess(coeffs, np.random.randn(tmls.base_dims, tmls.spline_order))

    
    # Foothold plan
    tmls.p = tmls.prog.NewContinuousVariables(
        tmls.num_legs, 3, 
        'p'
    )
    
    # Stability constraints slack variables
    tmls.epsilon = tmls.prog.NewContinuousVariables(
        num_phases, 
        'epsilon'
    )

# STATE UPDATE
