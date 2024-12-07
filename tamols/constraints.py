import numpy as np
from tamols import TAMOLSState
from helpers import (
    get_num_contacts, get_stance_feet, get_contact_pairs, 
    evaluate_spline_position, evaluate_spline_acceleration, 
    evaluate_angular_momentum_derivative, evaluate_height_at_xy, 
    determinant
)

def add_initial_constraints(tmls: TAMOLSState, log: bool = False):
    """Add initial and junction constraints on the spline coefficients"""
    print("Adding initial constraints...")
    if tmls.base_pose is None or tmls.p_meas is None or tmls.base_vel is None:
        raise ValueError("Initial state not set")
    
    # SPLINE INITIAL CONSTRAINTS
    a0 = tmls.spline_coeffs[0] 
    T0 = tmls.phase_durations[0]
    for dim in range(tmls.base_dims):
        tmls.prog.AddLinearConstraint(
            a0[dim,0] == tmls.base_pose[dim]
        )
        tmls.prog.AddLinearConstraint(
            a0[dim,1] / T0 == tmls.base_vel[dim]
        )

    # SPLINE JUNCTION CONSTRAINTS
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

    # NOTE: check if there need to be any intial conditions on feet

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

        N = get_num_contacts(tmls, phase)
        stance_feet = get_stance_feet(tmls, phase) # returns the indices of the feet in contact (but does not account for p_meas vs p)
        p_alr_at_des_pos = tmls.gait_pattern['at_des_position'][phase]

        eps = tmls.epsilon[phase]
        
        for tau in tmls.taus_to_check:
            p_B = evaluate_spline_position(tmls, a_k, tau)[:3]
            a_B = evaluate_spline_acceleration(tmls, a_k, tau, T_k)
            L_dot_B = evaluate_angular_momentum_derivative(tmls, tau)
            
            if N > 0: # Eq 17a: Friction cone constraint - FIXED
                proj = I_3 - np.outer(e_z, e_z)
                proj_acc = proj @ a_B

                print(f"adding N={N} constraint 17a")
                tmls.prog.AddConstraint(
                    (mu * e_z.dot(a_B))**2 >= (1 + eps)**2 * proj_acc.dot(proj_acc)
                )
 
            if N >= 3: # Eq 17b: Multiple contact GIAC constraints
                for i, j in get_contact_pairs(tmls, stance_feet):

                    p_i = tmls.p[i] if p_alr_at_des_pos[i] else tmls.p_meas[i]
                    p_j = tmls.p[j] if p_alr_at_des_pos[j] else tmls.p_meas[j]
                    p_ij = p_j - p_i
                    
                    print(f"adding N={N} constraint 17b")
                    tmls.prog.AddConstraint(
                        m * determinant(p_ij, p_B - p_i, a_B) <= 
                        (1 + eps) * p_ij.dot(L_dot_B)
                    )
                
                    
            elif N == 2: 
                # Eq 17c,d: Double support constraints
                i, j = stance_feet
                p_i = tmls.p[i] if p_alr_at_des_pos[i] else tmls.p_meas[i]
                p_j = tmls.p[j] if p_alr_at_des_pos[j] else tmls.p_meas[j]
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

            # SKIP N < 2 CASES FOR NOW

def add_kinematic_constraints(tmls: TAMOLSState):
    """Max distance between hip and foot"""
    print("Adding kinematic constraints...")

    for leg_idx in range(tmls.num_legs):
        for phase_idx, at_des_pos in enumerate(tmls.gait_pattern['at_des_position']):
            if at_des_pos[leg_idx]:
                for tau in tmls.taus_to_check:
                    spline_pos = evaluate_spline_position(tmls, tmls.spline_coeffs[phase_idx], tau)[0:3]
                    hip_location = spline_pos + tmls.hip_offsets[leg_idx]
                    diff = tmls.p[leg_idx] - hip_location
                    # tmls.prog.AddConstraint(
                    #     np.linalg.norm(diff) <= tmls.l_max
                    # )

                    tmls.prog.AddQuadraticConstraint(
                        diff.dot(diff),  # Quadratic expression
                        0, # quadratic lower bound is a non-convex constraint SKIP for now?
                        tmls.l_max**2
                    )


def add_giac_constraints(tmls: TAMOLSState):
    """Enforce feet form a convex polygon"""
    print("Adding GIAC constraints...")

    # Get the final foot locations
    # p_1 = tmls.p_meas[0]
    # p_2 = tmls.p_meas[1]
    # p_3 = tmls.p_meas[2]
    # p_4 = tmls.p_meas[3]
    p_1 = tmls.p[0]
    p_2 = tmls.p[1]
    p_3 = tmls.p[2]
    p_4 = tmls.p[3]

    # Constraint (p12 × p13) z-component ≤ 0 for p_meas
    p_12 = p_2 - p_1
    p_13 = p_3 - p_1
    tmls.prog.AddConstraint(p_12[0] * p_13[1] - p_12[1] * p_13[0] <= 0)

    # Constraint (p14 × p13) z-component ≤ 0 for p_meas
    p_14 = p_4 - p_1
    tmls.prog.AddConstraint(p_14[0] * p_13[1] - p_14[1] * p_13[0] <= 0)

    # Constraint (p24 × p23) z-component ≤ 0 for p_meas
    p_23 = p_3 - p_2
    p_24 = p_4 - p_2
    tmls.prog.AddConstraint(p_24[0] * p_23[1] - p_24[1] * p_23[0] <= 0)

    # Constraint (p24 × p21) z-component ≤ 0 for p_meas
    p_21 = p_1 - p_2
    tmls.prog.AddConstraint(p_24[0] * p_21[1] - p_24[1] * p_21[0] <= 0)

