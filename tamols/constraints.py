import numpy as np
from pydrake.all import (
    cos, sin
)
from tamols import TAMOLSState
from helpers import (
    get_num_contacts, get_stance_feet, get_contact_pairs, 
    evaluate_spline_position, evaluate_spline_acceleration, 
    evaluate_angular_momentum_derivative, evaluate_height_at_xy, 
    determinant
)



# NOTE: very fast even w for loops
def add_initial_constraints(tmls: TAMOLSState, log: bool = False):
    """Add initial state constraints for base pose, velocity, and foot positions"""
    print("Adding initial constraints...")

    if tmls.base_pose is None or tmls.p_meas is None or tmls.base_vel is None:
        raise ValueError("Initial state not set")
    
    a0 = tmls.spline_coeffs[0]  # First phase coefficients
    T0 = tmls.phase_durations[0]  # Duration of first phase
    
    # Initial constraints
    for dim in range(tmls.base_dims):
        tmls.prog.AddLinearConstraint(
            a0[dim,0] == tmls.base_pose[dim]
        )
        tmls.prog.AddLinearConstraint(
            a0[dim,1] / T0 == tmls.base_vel[dim]
        )
    
    # for leg_idx in range(tmls.num_legs):
    #     for dim in range(3):
    #         tmls.prog.AddLinearConstraint(
    #             tmls.p[leg_idx,dim] == tmls.p_meas[leg_idx,dim]
    #         )


    # NOTE: HARD CODING FINAL POSITIONS
    for leg_idx, pos in enumerate([[0.4, 0.1, 0], [0.4, -0.1, 0], [0, 0.1, 0], [0, -0.1, 0]]):
        for dim in range(3):
            tmls.prog.AddLinearConstraint(tmls.p[leg_idx, dim] == pos[dim])
    
    # Spline constraints
    num_phases = len(tmls.phase_durations)
    for phase in range(num_phases - 1):
        ak = tmls.spline_coeffs[phase]      # Current phase
        ak1 = tmls.spline_coeffs[phase+1]   # Next phase
        Tk = tmls.phase_durations[phase]    # Current phase duration
        Tk1 = tmls.phase_durations[phase+1] # Next phase duration
        
        for dim in range(tmls.base_dims):
            # Position continuity: evaluate end of current = start of next
            pos_k = sum(ak[dim,i] for i in range(tmls.spline_order))  # τ = 1
            pos_k1 = ak1[dim,0]                                       # τ = 0
            tmls.prog.AddLinearConstraint(pos_k == pos_k1)
            
            # Velocity continuity
            vel_k = sum(i * ak[dim,i] / Tk for i in range(1, tmls.spline_order))   # τ = 1
            vel_k1 = ak1[dim,1] / Tk1                                              # τ = 0
            tmls.prog.AddLinearConstraint(vel_k == vel_k1)

# NOTE: Check that the math here is accurate
def add_dynamics_constraints(tmls: TAMOLSState):
    """Add GIAC stability constraints at mid-phase and end of each phase"""
    print("Adding dynamics constraints...")

    # Constants
    num_phases = len(tmls.phase_durations)
    
    e_z = np.array([0., 0., 1.])
    I_3 = np.eye(3)

    mu = tmls.mu
    m = tmls.mass

    for phase in range(num_phases):
        a_k = tmls.spline_coeffs[phase]
        T_k = tmls.phase_durations[phase]

        p_at_des_pos = tmls.gait_pattern['at_des_position'][phase]
        
        N = get_num_contacts(tmls, phase)
        stance_feet = get_stance_feet(tmls, phase) # returns the indices of the feet in contact (but does not account for p_meas vs p)
        # TODO: add logic here to handle p_meas vs p

        eps = tmls.epsilon[phase]
        
        # Evaluate at pre-set waypoints
        for tau in tmls.taus_to_check:
            p_B = evaluate_spline_position(tmls, a_k, tau)[:3]
            a_B = evaluate_spline_acceleration(tmls, a_k, tau, T_k)
            L_dot_B = evaluate_angular_momentum_derivative(tmls, tau)
            
            if N > 0: # Eq 17a: Friction cone constraint - FIXED
                proj = I_3 - np.outer(e_z, e_z)
                proj_acc = proj @ a_B
                tmls.prog.AddConstraint(
                    (mu * e_z.dot(a_B))**2 >= (1 + eps)**2 * proj_acc.dot(proj_acc)
                )
            

            # NOTE IN THESE DYNAMICS CONSTRAINTS WE MUST BE CAREFUL ABOUT 
            # WHETHER THE LEG POSITION IS FROM P or P_MEAS
            
 
            if N >= 3: # Eq 17b: Multiple contact GIAC constraints
                for i, j in get_contact_pairs(tmls, stance_feet):

                    p_i = tmls.p[i] if p_at_des_pos[i] else tmls.p_meas[i]
                    p_j = tmls.p[j] if p_at_des_pos[j] else tmls.p_meas[j]
                    p_ij = p_j - p_i
                    
                    print(f"adding N={N} constraint 17b")
                    tmls.prog.AddConstraint(
                        m * determinant(p_ij, p_B - p_i, a_B) <= 
                        (1 + eps) * p_ij.dot(L_dot_B)
                    )
                
                    
            elif N == 2: 
                # Eq 17c,d: Double support constraints
                i, j = stance_feet
                p_i = tmls.p[i] if p_at_des_pos[i] else tmls.p_meas[i]
                p_j = tmls.p[j] if p_at_des_pos[j] else tmls.p_meas[j]
                p_ij = p_j - p_i
                
                # 17c: Equality constraint
                print(f"adding N={N} constraint 17c")
                tmls.prog.AddConstraint(
                    m * determinant(p_ij, p_B - p_i, a_B) == 
                    p_ij.dot(L_dot_B)
                )
                
                # 17d: Moment constraint
                print(f"adding N={N} constraint 17d")
                M_i = m * np.cross(p_B - p_i, a_B) - L_dot_B
                tmls.prog.AddConstraint(
                    determinant(e_z, p_ij, M_i) >= 0
                )

            # NOTE: SKIP - USING TROT GAIT (NEVER 0 CONTACTS)
            # elif N == 1:
            #     # Eq 17e: Single support constraint
            #     i = stance_feet[0]
            #     p_i = tmls.p[i] if p_at_des_pos[i] else tmls.p_meas[i]
                
            #     tmls.prog.AddConstraint(
            #         m * np.cross(p_B - p_i, a_B) == L_dot_B
            #     )
            
            # else:  # N == 0
            #     # Eq 17f: Flight phase constraints
            #     tmls.prog.AddConstraint(a_B == np.zeros(3))
            #     tmls.prog.AddConstraint(L_dot_B == np.zeros(3))

def add_kinematic_constraints(tmls: TAMOLSState):
    """
    Add kinematic feasibility constraints:
    - Leg length limits
    - Workspace constraints 
    - Collision avoidance (Not implemented)
    """
    print("Adding kinematic constraints...")

    hip_offsets = tmls.hip_offsets
    l_min = tmls.l_min
    l_max = tmls.l_max

    for leg_idx in range(tmls.num_legs):
        p_foot = tmls.p[leg_idx]
        hip_offset_body = hip_offsets[leg_idx]
        
        # Check constraints at collocation points
        for phase in range(len(tmls.phase_durations)):
            a_k = tmls.spline_coeffs[phase]
            
            for tau in tmls.taus_to_check:
                pos_B = evaluate_spline_position(tmls, a_k, tau)
                rpy_B = evaluate_spline_position(tmls, a_k, tau)[3:6]
                
                # Convert Euler angles to rotation matrix

                # Instead of using RollPitchYaw, directly compute rotation 
                # matrix using optimization variables
                cr = cos(rpy_B[0])  # roll
                sr = sin(rpy_B[0])
                cp = cos(rpy_B[1])  # pitch
                sp = sin(rpy_B[1])
                cy = cos(rpy_B[2])  # yaw
                sy = sin(rpy_B[2])
                
                # Construct rotation matrix directly
                R_B = np.array([
                    [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                    [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                    [-sp, cp*sr, cp*cr]
                ])
                
                # Compute hip position in world frame
                hip_pos_world = pos_B[:3] + R_B @ hip_offset_body
                
                # Vector from hip to foot
                leg_vector = p_foot - hip_pos_world
                
                # Leg length constraints
                leg_length = np.linalg.norm(leg_vector)
                tmls.prog.AddConstraint(
                    leg_length >= l_min
                )
                tmls.prog.AddConstraint(
                    leg_length <= l_max
                )
                
                # Workspace constraints in world frame
                tmls.prog.AddConstraint(
                    p_foot[2] <= pos_B[2] - .04  # Foot below bases
                )
                
                # Additional workspace bounds
                workspace_bounds = {
                    'x': [-0.3, 0.3],
                    'y': [-0.2, 0.2],
                    'z': [-0.5, 0]
                }
                
                for i, (min_val, max_val) in enumerate([
                    workspace_bounds['x'],
                    workspace_bounds['y'], 
                    workspace_bounds['z']
                ]):
                    tmls.prog.AddConstraint(
                        leg_vector[i] >= min_val
                    )
                    tmls.prog.AddConstraint(
                        leg_vector[i] <= max_val
                    )
                
                # TODO: tmls-collision avoidance between legs
                # NOTE: THIS IS FORMULATED AS A QUADRATIC BARRIER COST THOUGH????
                for other_leg_idx in range(leg_idx + 1, tmls.num_legs):
                    other_p_foot = tmls.p[other_leg_idx]
                    min_distance = 0.1  # Minimum distance between feet
                    
                    # Distance in xy-plane
                    foot_distance = (p_foot[:2] - other_p_foot[:2]).dot(
                        p_foot[:2] - other_p_foot[:2]
                    )
                    

                    # NOTE:gives warning about definiteness of this matrix ()
                    # tmls.prog.AddConstraint(
                    #     foot_distance >= min_distance**2,
                    # )

def add_gait_constraints(tmls: TAMOLSState):
    """Add constraints based on gait pattern timing and contact states"""
    print("Adding gait constraints...")

    phase_times = tmls.gait_pattern['phase_timing']
    contact_states = tmls.gait_pattern['contact_states']
    
    for phase_idx in range(len(tmls.phase_durations)):
        phase_contacts = contact_states[phase_idx]
        
        for leg_idx in range(tmls.num_legs):
            p_foot = tmls.p[leg_idx]
            
            if phase_contacts[leg_idx]:  # Leg should be in contact
                # Ground contact constraint
                tmls.prog.AddConstraint(
                    p_foot[2] == evaluate_height_at_xy(tmls, p_foot[0], p_foot[1])
                )
                
            # NOTE THIS CONSTRAINT CAUSES ISSUES GOING TO SKIP FOR NOW
            # else:  # Leg should be in swing
            #     # Minimum ground clearance constraint
            #     min_clearance = 0.05
            #     tmls.prog.AddConstraint(
            #         p_foot[2] >= evaluate_height_at_xy(p_foot[0], p_foot[1]) + min_clearance
            #     )
                
        # Ensure timing consistency between phases
        if phase_idx < len(phase_times) - 1:
            tmls.prog.AddConstraint(
                phase_times[phase_idx + 1] >= phase_times[phase_idx]
            )
                    
        # Ensure timing consistency between phases
        for i in range(len(phase_times)-1):
            tmls.prog.AddConstraint(
                phase_times[i+1] >= phase_times[i]
            )
