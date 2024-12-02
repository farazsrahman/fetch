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

def add_costs(tmls: TAMOLSState):
    """Add all optimization cost terms"""
    print("\nAdding costs...")

    # 1. Robustness margin (using epsilon from dynamics constraints)
    add_robustness_cost(tmls)
    
    # 2. Footholds on ground
    add_foothold_ground_cost(tmls)
    
    # 3. Leg collision avoidance
    add_leg_collision_cost(tmls)
    
    # 4. Nominal kinematics
    add_nominal_kinematics_cost(tmls)
    
    # # 5. Base pose alignment with terrain
    # add_base_alignment_cost(tmls)
    
    # # 6. Edge avoidance
    # add_edge_avoidance_cost(tmls)
    
    # # 7. Previous solution tracking (if available)
    # if hasattr(tmls, 'previous_solution'):
    #     add_previous_solution_cost(tmls)
        
    # # 8. Reference trajectory tracking
    # add_tracking_cost(tmls)
    
    # # 9. Smoothness (minimize angular momentum change)
    # add_smoothness_cost(tmls)

def add_robustness_cost(tmls: TAMOLSState):
    """Cost to maximize stability margin through epsilon"""
    print("Adding robustness cost...")

    w = tmls.weights['robustness_margin']
    eps = tmls.epsilon
    tmls.prog.AddQuadraticCost(w * np.sum(eps**2))

def add_foothold_ground_cost(tmls: TAMOLSState):
    """Cost to keep feet on the terrain"""
    print("Adding foot on ground cost...")

    w = tmls.weights['footholds_on_ground']
    for i in range(tmls.num_legs):
        p_foot = tmls.p[i]
        target_height = evaluate_height_at_xy(tmls, p_foot[0], p_foot[1])
        height_diff = target_height - p_foot[2]
        tmls.prog.AddQuadraticCost(w * height_diff**2)

def add_leg_collision_cost(tmls: TAMOLSState):
    """Cost to avoid collisions between legs"""
    # relaxed to only put a quadratic cost
    print("Adding leg collision cost...")

    w = tmls.weights['leg_collision_avoidance']
    for i in range(tmls.num_legs):
        for j in range(i+1, tmls.num_legs):
            p_i = tmls.p[i]
            p_j = tmls.p[j]
            
            # Distance in xy-plane
            xy_dist = (p_i[:2] - p_j[:2]).dot(p_i[:2] - p_j[:2])
            
            # Reformulate the cost as a quadratic barrier on min distance
            min_dist_sq = tmls.min_foot_distance**2
            barrier = min_dist_sq - xy_dist
            
            # use pydrake if_then_else to enforce barrier constraint
            cost_term = if_then_else(barrier > 0, w * barrier**2, 0)
            tmls.prog.AddCost(cost_term)
        
def add_nominal_kinematics_cost(tmls: TAMOLSState):

    print("\n\n\n")
    print("Adding nominal kinematics cost...")

    w = tmls.weights['nominal_kinematics']
    l_des = np.array([0, 0, tmls.desired_height])
    
    for phase in range(len(tmls.gait_pattern['phase_timing'])-1):
        # Get base pose at mid-stance
        a_k = tmls.spline_coeffs[phase]
        pos_B = evaluate_spline_position(tmls, a_k, 0.5)
        rpy_B = pos_B[3:6]

        print(f"\n\nSpline coeffs @ phase {phase}:")
        print(f"pos_B: {pos_B}\n\n")
        print(f"rpy_B: {rpy_B}\n\n")

        # break
        phi = rpy_B[0]
        theta = rpy_B[1]
        psi = rpy_B[2]

        # Create the rotation matrix
        R_B = np.array([
            [cos(theta) * cos(psi),
            sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi),
            cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)],
            [cos(theta) * sin(psi),
            sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi),
            cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)],
            [-sin(theta),
            sin(phi) * cos(theta),
            cos(phi) * cos(theta)]
        ])
        
        print("R_B type: ", type(R_B))
        print(f"R_B: {R_B}\n\n")

        print("RotationMatrix.ProjectToRotationMatrix: ", RotationMatrix.ProjectToRotationMatrix(R_B))
        
        

        # TODO: FIGURE OUT HOW TO GET AROUND ROTATION MATRIX ISSUES
        
        # for leg_idx in range(tmls.num_legs):
        #     p_foot = tmls.p[leg_idx]
        #     # Use matrix multiplication instead of @ operator for symbolic expressions
        #     hip_offset_rotated = R_B.multiply(tmls.hip_offsets[leg_idx])
        #     p_hip = pos_B[:3] + hip_offset_rotated
        #     error = p_foot - p_hip - l_des
            
        #     # For symbolic expressions, we need to use dot product explicitly
        #     cost = error[0] * error[0] + error[1] * error[1] + error[2] * error[2]
        #     tmls.prog.AddQuadraticCost(w * cost)

        break
            

def add_base_alignment_cost(tmls: TAMOLSState):
    """Cost to align base with terrain"""
    print("Adding base alignment cost...")

    w = tmls.weights['base_pose_alignment']
    
    for phase in range(len(tmls.gait_pattern['phase_timing'])):
        Tk = tmls.gait_pattern['phase_timing'][phase + 1] - tmls.gait_pattern['phase_timing'][phase]
        a_k = tmls.spline_coeffs[phase]
        
        # Sample points along trajectory
        for tau in tmls.taus_to_check:
            pos_B = evaluate_spline_position(a_k, tau)
            rpy_B = pos_B[3:6]
            
            # Compute rotation matrix directly
            cr = cos(rpy_B[0])
            sr = sin(rpy_B[0])
            cp = cos(rpy_B[1])
            sp = sin(rpy_B[1])
            cy = cos(rpy_B[2])
            sy = sin(rpy_B[2])
            
            R_B = np.array([
                [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                [-sp, cp*sr, cp*cr]
            ])
            
            for i in range(4):
                r_i = tmls.hip_offsets[i]
                p_hip = pos_B[:3] + R_B @ r_i
                h_des = evaluate_height_at_xy(p_hip[0], p_hip[1])
                error = p_hip[2] - h_des
                tmls.prog.AddQuadraticCost(w * Tk * error**2)

def add_edge_avoidance_cost(tmls: TAMOLSState):
    """Cost to avoid terrain edges"""
    print("Adding edge avoidance cost...")

    w = tmls.weights['edge_avoidance']
    for i in range(tmls.num_legs):
        p_foot = tmls.p[i]
        # Assuming these functions return numerical gradients or expressions
        # without requiring boolean operations
        grad_h = evaluate_height_gradient(p_foot[0], p_foot[1])
        grad_hs1 = evaluate_smoothed_height_gradient(p_foot[0], p_foot[1])
        tmls.prog.AddQuadraticCost(w * (grad_h.dot(grad_h) + grad_hs1.dot(grad_hs1)))

def add_previous_solution_cost(tmls: TAMOLSState):
    """Cost to stay close to previous solution"""
    print("Adding prev soln cost...")

    w = tmls.weights['previous_solution']
    for i in range(tmls.num_legs):
        error = tmls.p[i] - tmls.previous_solution['footholds'][i]
        tmls.prog.AddQuadraticCost(w * error.dot(error))

def add_tracking_cost(tmls: TAMOLSState):
    """Cost to track reference trajectory"""
    print("Adding tracking cost...")

    w = tmls.weights['tracking']
    
    for phase in range(len(tmls.gait_pattern['phase_timing'])):
        Tk = tmls.gait_pattern['phase_timing'][phase + 1] - tmls.gait_pattern['phase_timing'][phase]
        a_k = tmls.spline_coeffs[phase]
        
        # Sample points along trajectory
        for tau in tmls.taus_to_check:
            # Get current state
            pos = evaluate_spline_position(a_k, tau)
            vel = evaluate_spline_velocity(a_k, tau, Tk)
            
            # Compute errors
            pos_error = pos - tmls.reference_trajectory['pos'](tau * Tk)
            vel_error = vel - tmls.reference_trajectory['vel'](tau * Tk)
            
            # Weight position/velocity errors differently for base vs angular terms 
            error = np.concatenate([
                pos_error[:3] / tmls.mass,  # Linear position
                vel_error[:3],              # Linear velocity
                pos_error[3:] @ tmls.inertia,  # Angular position
                vel_error[3:]                  # Angular velocity
            ])
            
            tmls.prog.AddQuadraticCost(w * Tk * error.dot(error))

def add_smoothness_cost(tmls: TAMOLSState):
    """Cost to minimize angular momentum changes"""
    print("Adding smoothness cost...")

    w = tmls.weights['smoothness']
    
    for phase in range(len(tmls.gait_pattern['phase_timing'])):
        Tk = tmls.gait_pattern['phase_timing'][phase + 1] - tmls.gait_pattern['phase_timing'][phase]
        a_k = tmls.spline_coeffs[phase]
        
        for tau in tmls.taus_to_check:
            L_dot = evaluate_angular_momentum_derivative(tau)
            tmls.prog.AddQuadraticCost(w * Tk * L_dot.dot(L_dot))

