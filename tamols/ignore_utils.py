import numpy as np
from pydrake.all import (
    MathematicalProgram, Solve, Variable,
    cos, sin
)
from pydrake.math import RollPitchYaw, RotationMatrix


class TAMOLS:
    def __init__(self, dt=0.01, horizon_steps=100, weights=None):
        """Initialize TAMOLS planner"""
        self.dt = dt
        self.horizon_steps = horizon_steps

        # Height Map
        self.height_map = None
        self.height_map_smoothed = None
        
        # State dimensions
        self.num_legs = 4
        self.spline_order = 5
        self.base_dims = 6  # [x,y,z,roll,pitch,yaw]

        # Physical parameters # TODO: MAKE SURE THESE ACTUALLY ALIGN WITH THE GO2
        self.mass = 1.0  # Robot mass
        self.mu = 0.7    # Friction coefficient
        self.inertia = np.diag([0.07, 0.26, 0.242])  # Example values, adjust for your robot

        self.hip_offsets = np.array([
            [ 0.2,  0.15, 0],  # Front Left
            [ 0.2, -0.15, 0],  # Front Right
            [-0.2,  0.15, 0],  # Back Left
            [-0.2, -0.15, 0],  # Back Right
        ])

        self.l_min = 0.2  # Minimum leg length
        self.l_max = 0.5  # Maximum leg length
            
        # Current state
        self.p_meas = None
        self.base_pose = None
        self.base_vel = None
        
        # Initialize optimization program
        self.taus_to_check = [0.5, 1.0] 
        self.prog = None
        self.vars = {}


        # Initialize cost weights
        if weights == None:
            self.weights = {
                'robustness_margin': 0.007,
                'footholds_on_ground': 1e4,
                'leg_collision_avoidance': 0.001,
                'nominal_kinematics': 7.0,
                'base_pose_alignment': 100.0,  # Multiplied by phase duration
                'edge_avoidance': 3.0,
                'previous_solution': 0.01,
                'tracking': 2.0,  # Multiplied by phase duration
                'smoothness': 0.001  # Multiplied by phase duration
            }


        # desidirata
        self.nominal_height = 0.4  # Desired standing height
        self.foot_radius = 0.02  # Radius of robot feet
        self.min_foot_distance = 0.1  # Minimum distance between feet
        self.desired_height = 0.4  # Desired height above ground

     # SETUP

    def setup_variables(self, gait_pattern):
        """Setup optimization variables based on gait phases"""
        self.prog = MathematicalProgram()
        
        # Setting up phase (why here)
        phase_times = gait_pattern['phase_timing']
        num_phases = len(phase_times) - 1  # Number of intervals between timestamps
        self.phase_durations = [
            phase_times[i+1] - phase_times[i] 
            for i in range(num_phases)
        ]

        # Spline coefficients
        self.vars['spline_coeffs'] = []
        for i in range(num_phases):
            self.vars['spline_coeffs'].append(self.prog.NewContinuousVariables(
                self.base_dims, self.spline_order, 
                f'a_{i}'
            ))
        
        # Foothold plan
        self.vars['p'] = self.prog.NewContinuousVariables(
            self.num_legs, 3, 
            'p'
        )
        
        # Stability constraints slack variables
        self.vars['epsilon'] = self.prog.NewContinuousVariables(
            num_phases, 
            'epsilon'
        )

    def set_reference_trajectory(self, reference_trajectory):
        """Set the reference trajectory functions"""
        self.reference_trajectory = reference_trajectory

    # CONSTRAINTS
    
    # NOTE: I feel as though some of these operations could definitely be vectorized
    def add_initial_constraints(self):
        """Add initial state constraints for base pose, velocity, and foot positions"""
        print("Adding initial constraints...")

        if self.base_pose is None or self.p_meas is None or self.base_vel is None:
            raise ValueError("Initial state not set")
        
        a0 = self.vars['spline_coeffs'][0]  # First phase coefficients
        T0 = self.phase_durations[0]         # Duration of first phase
        
        # Initial base pose constraint
        for dim in range(self.base_dims):
            self.prog.AddLinearConstraint(
                a0[dim,0] == self.base_pose[dim]
            )
            self.prog.AddLinearConstraint(
                a0[dim,1] / T0 == self.base_vel[dim]
            )
        
        # Initial foot positions constraint
        for leg_idx in range(self.num_legs):
            for dim in range(3):
                self.prog.AddLinearConstraint(
                    self.vars['p'][leg_idx,dim] == self.p_meas[leg_idx,dim]
                )
        
        # Spline junction constraints
        num_phases = len(self.phase_durations)
        for phase in range(num_phases - 1):
            # Get spline coefficients for adjacent phases
            ak = self.vars['spline_coeffs'][phase]      # Current phase
            ak1 = self.vars['spline_coeffs'][phase+1]   # Next phase
            Tk = self.phase_durations[phase]            # Current phase duration
            Tk1 = self.phase_durations[phase+1]         # Next phase duration
            
            for dim in range(self.base_dims):
                # Position continuity: evaluate end of current = start of next
                pos_k = sum(ak[dim,i] for i in range(self.spline_order))  # τ = 1
                pos_k1 = ak1[dim,0]                                       # τ = 0
                self.prog.AddLinearConstraint(pos_k == pos_k1)
                
                # Velocity continuity
                vel_k = sum(i * ak[dim,i] / Tk for i in range(1, self.spline_order))   # τ = 1
                vel_k1 = ak1[dim,1] / Tk1                                              # τ = 0
                self.prog.AddLinearConstraint(vel_k == vel_k1)

    # NOTE: Check that the math here is accurate
    def add_dynamics_constraints(self):
        """Add GIAC stability constraints at mid-phase and end of each phase"""
        print("Adding dynamics constraints...")

        
        # Constants
        num_phases = len(self.phase_durations)
        
        e_z = np.array([0., 0., 1.])
        I_3 = np.eye(3)

        mu = self.mu
        m = self.mass

        
        for phase in range(num_phases):
            a_k = self.vars['spline_coeffs'][phase]
            T_k = self.phase_durations[phase]
            
            N = self.get_num_contacts(phase)
            stance_feet = self.get_stance_feet(phase)
            eps = self.vars['epsilon'][phase]
            
            # Evaluate at pre-set waypoints
            for tau in self.taus_to_check:
                p_B = self.evaluate_spline_position(a_k, tau)[:3] # since this function returns the full spline w rotaiton vectors
                a_B = self.evaluate_spline_acceleration(a_k, tau, T_k)
                L_dot_B = self.evaluate_angular_momentum_derivative(tau)
                

                if N > 0: # Eq 17a: Friction cone constraint - FIXED
                    proj = I_3 - np.outer(e_z, e_z)
                    proj_acc = proj @ a_B
                    self.prog.AddConstraint(
                        (mu * e_z.dot(a_B))**2 >= (1 + eps)**2 * proj_acc.dot(proj_acc)
                    )
                
                if N >= 3: # Eq 17b: Multiple contact GIAC constraints
                    for i, j in self.get_contact_pairs(stance_feet):
                        p_i = self.vars['p'][i]
                        p_j = self.vars['p'][j]
                        p_ij = p_j - p_i
                        
                        self.prog.AddConstraint(
                            m * self.determinant(p_ij, p_B - p_i, a_B) <= 
                            (1 + eps) * p_ij.dot(L_dot_B)
                        )
                        
                elif N == 2: 
                    # Eq 17c,d: Double support constraints
                    i, j = stance_feet
                    p_i = self.vars['p'][i]
                    p_j = self.vars['p'][j]
                    p_ij = p_j - p_i
                    
                    # 17c: Equality constraint
                    self.prog.AddConstraint(
                        m * self.determinant(p_ij, p_B - p_i, a_B) == 
                        p_ij.dot(L_dot_B)
                    )
                    
                    # 17d: Moment constraint
                    M_i = m * np.cross(p_B - p_i, a_B) - L_dot_B
                    self.prog.AddConstraint(
                        self.determinant(e_z, p_ij, M_i) >= 0
                    )
                    
                elif N == 1:
                    # Eq 17e: Single support constraint
                    i = stance_feet[0]
                    p_i = self.vars['p'][i]
                    
                    self.prog.AddConstraint(
                        m * np.cross(p_B - p_i, a_B) == L_dot_B
                    )
                    
                else:  # N == 0
                    # Eq 17f: Flight phase constraints
                    self.prog.AddConstraint(a_B == np.zeros(3))
                    self.prog.AddConstraint(L_dot_B == np.zeros(3))

    def add_kinematic_constraints(self):
        """
        Add kinematic feasibility constraints:
        - Leg length limits
        - Workspace constraints 
        - Collision avoidance
        """
        print("Adding kinematic constraints...")

        hip_offsets = self.hip_offsets
        l_min = self.l_min
        l_max = self.l_max

        for leg_idx in range(self.num_legs):
            p_foot = self.vars['p'][leg_idx]
            hip_offset_body = hip_offsets[leg_idx]
            
            # Check constraints at collocation points
            for phase in range(len(self.phase_durations)):
                a_k = self.vars['spline_coeffs'][phase]
                
                for tau in self.taus_to_check:
                    pos_B = self.evaluate_spline_position(a_k, tau)
                    rpy_B = self.evaluate_spline_position(a_k, tau)[3:6]
                    
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
                    self.prog.AddConstraint(
                        leg_length >= l_min
                    )
                    self.prog.AddConstraint(
                        leg_length <= l_max
                    )
                    
                    # Workspace constraints in world frame
                    self.prog.AddConstraint(
                        p_foot[2] <= pos_B[2]  # Foot below base
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
                        self.prog.AddConstraint(
                            leg_vector[i] >= min_val
                        )
                        self.prog.AddConstraint(
                            leg_vector[i] <= max_val
                        )
                    
                    # Self-collision avoidance between legs
                    for other_leg_idx in range(leg_idx + 1, self.num_legs):
                        other_p_foot = self.vars['p'][other_leg_idx]
                        min_distance = 0.1  # Minimum distance between feet
                        
                        # Distance in xy-plane
                        foot_distance = (p_foot[:2] - other_p_foot[:2]).dot(
                            p_foot[:2] - other_p_foot[:2]
                        )
                        
                        self.prog.AddConstraint(
                            foot_distance >= min_distance**2
                        )

    def add_gait_constraints(self, gait_pattern):
        """Add constraints based on gait pattern timing and contact states"""
        print("Adding gait constraints...")

        phase_times = gait_pattern['phase_timing']
        contact_states = gait_pattern['contact_states']
        
        for phase_idx in range(len(self.phase_durations)):
            phase_contacts = contact_states[phase_idx]
            
            for leg_idx in range(self.num_legs):
                p_foot = self.vars['p'][leg_idx]
                
                if phase_contacts[leg_idx]:  # Leg should be in contact
                    # Ground contact constraint
                    self.prog.AddConstraint(
                        p_foot[2] == self.evaluate_height_at_xy(p_foot[0], p_foot[1])
                    )
                    
                # NOTE THIS CONSTRAINT CAUSES ISSUES GOING TO SKIP FOR NOW
                # else:  # Leg should be in swing
                #     # Minimum ground clearance constraint
                #     min_clearance = 0.05
                #     self.prog.AddConstraint(
                #         p_foot[2] >= self.evaluate_height_at_xy(p_foot[0], p_foot[1]) + min_clearance
                #     )
                    
            # Ensure timing consistency between phases
            if phase_idx < len(phase_times) - 1:
                self.prog.AddConstraint(
                    phase_times[phase_idx + 1] >= phase_times[phase_idx]
                )
                        
            # Ensure timing consistency between phases
            for i in range(len(phase_times)-1):
                self.prog.AddConstraint(
                    phase_times[i+1] >= phase_times[i]
                )


    # COST FUNCTIONS

    def add_costs(self, prog):
        """Add all optimization cost terms"""
        print("Adding costs...")
        # 1. Robustness margin (using epsilon from dynamics constraints)
        self.add_robustness_cost(prog)
        
        # 2. Footholds on ground
        self.add_foothold_ground_cost(prog)
        
        # # 3. Leg collision avoidance
        # self.add_leg_collision_cost(prog)
        
        # # 4. Nominal kinematics
        # self.add_nominal_kinematics_cost(prog)
        
        # # 5. Base pose alignment with terrain
        # self.add_base_alignment_cost(prog)
        
        # # 6. Edge avoidance
        # self.add_edge_avoidance_cost(prog)
        
        # 7. Previous solution tracking (if available)
        if hasattr(self, 'previous_solution'):
            self.add_previous_solution_cost(prog)
            
        # 8. Reference trajectory tracking
        self.add_tracking_cost(prog)
        
        # 9. Smoothness (minimize angular momentum change)
        self.add_smoothness_cost(prog)

    def add_robustness_cost(self, prog):
        """Cost to maximize stability margin through epsilon"""
        print("Adding robustness cost...")

        w = self.weights['robustness_margin']
        eps = self.vars['epsilon']
        prog.AddQuadraticCost(w * np.sum(eps**2))

    def add_foothold_ground_cost(self, prog):
        """Cost to keep feet on the terrain"""
        print("Adding foot on ground cost...")

        w = self.weights['footholds_on_ground']
        for i in range(self.num_legs):
            p_foot = self.vars['p'][i]
            target_height = self.evaluate_height_at_xy(p_foot[0], p_foot[1])
            height_diff = target_height - p_foot[2]
            prog.AddQuadraticCost(w * height_diff**2)

    def add_leg_collision_cost(self, prog):
        """Cost to avoid collisions between legs"""
        # relaxed to only put a quadratic cost
        print("Adding leg collision cost...")

        w = self.weights['leg_collision_avoidance']
        for i in range(self.num_legs):
            for j in range(i+1, self.num_legs):
                p_i = self.vars['p'][i]
                p_j = self.vars['p'][j]
                
                # Distance in xy-plane
                xy_dist = (p_i[:2] - p_j[:2]).dot(p_i[:2] - p_j[:2])
                
                # Reformulate the cost as a quadratic barrier on min distance
                min_dist_sq = self.min_foot_distance**2
                barrier = min_dist_sq - xy_dist
                prog.AddQuadraticCost(w * barrier**2 * (barrier > 0))
            
    def add_nominal_kinematics_cost(self, prog):
        """Cost to maintain nominal leg configuration"""
        print("Adding nominal kinematics cost...")

        w = self.weights['nominal_kinematics']
        l_des = np.array([0, 0, self.desired_height])
        
        for phase in range(len(self.phase_durations)):
            # Get base pose at mid-stance
            a_k = self.vars['spline_coeffs'][phase]
            pos_B = self.evaluate_spline_position(a_k, 0.5)
            rpy_B = pos_B[3:6]
            
            # Compute rotation matrix without boolean operations
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
            
            for leg_idx in range(self.num_legs):
                p_foot = self.vars['p'][leg_idx]
                p_hip = pos_B[:3] + R_B @ self.hip_offsets[leg_idx]
                error = p_foot - p_hip - l_des
                prog.AddQuadraticCost(w * error.dot(error))

    def add_base_alignment_cost(self, prog):
        """Cost to align base with terrain"""
        print("Adding base alignment cost...")

        w = self.weights['base_pose_alignment']
        
        for phase in range(len(self.phase_durations)):
            Tk = self.phase_durations[phase]
            a_k = self.vars['spline_coeffs'][phase]
            
            # Sample points along trajectory
            for tau in self.taus_to_check:
                pos_B = self.evaluate_spline_position(a_k, tau)
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
                    r_i = self.hip_offsets[i]
                    p_hip = pos_B[:3] + R_B @ r_i
                    h_des = self.evaluate_height_at_xy(p_hip[0], p_hip[1])
                    error = p_hip[2] - h_des
                    prog.AddQuadraticCost(w * Tk * error**2)

    def add_edge_avoidance_cost(self, prog):
        """Cost to avoid terrain edges"""
        print("Adding edge avoidance cost...")

        w = self.weights['edge_avoidance']
        for i in range(self.num_legs):
            p_foot = self.vars['p'][i]
            # Assuming these functions return numerical gradients or expressions
            # without requiring boolean operations
            grad_h = self.evaluate_height_gradient(p_foot[0], p_foot[1])
            grad_hs1 = self.evaluate_smoothed_height_gradient(p_foot[0], p_foot[1])
            prog.AddQuadraticCost(w * (grad_h.dot(grad_h) + grad_hs1.dot(grad_hs1)))

    def add_previous_solution_cost(self, prog):
        """Cost to stay close to previous solution"""
        print("Adding prev soln cost...")

        w = self.weights['previous_solution']
        for i in range(self.num_legs):
            error = self.vars['p'][i] - self.previous_solution['footholds'][i]
            prog.AddQuadraticCost(w * error.dot(error))

    def add_tracking_cost(self, prog):
        """Cost to track reference trajectory"""
        print("Adding tracking cost...")

        w = self.weights['tracking']
        
        for phase in range(len(self.phase_durations)):
            Tk = self.phase_durations[phase]
            a_k = self.vars['spline_coeffs'][phase]
            
            # Sample points along trajectory
            for tau in self.taus_to_check:
                # Get current state
                pos = self.evaluate_spline_position(a_k, tau)
                vel = self.evaluate_spline_velocity(a_k, tau, Tk)
                
                # Compute errors
                pos_error = pos - self.reference_trajectory['pos'](tau * Tk)
                vel_error = vel - self.reference_trajectory['vel'](tau * Tk)
                
                # Weight position/velocity errors differently for base vs angular terms 
                error = np.concatenate([
                    pos_error[:3] / self.mass,  # Linear position
                    vel_error[:3],              # Linear velocity
                    pos_error[3:] @ self.inertia,  # Angular position
                    vel_error[3:]                  # Angular velocity
                ])
                
                prog.AddQuadraticCost(w * Tk * error.dot(error))

    def add_smoothness_cost(self, prog):
        """Cost to minimize angular momentum changes"""
        print("Adding smoothness cost...")


        w = self.weights['smoothness']
        
        for phase in range(len(self.phase_durations)):
            Tk = self.phase_durations[phase]
            a_k = self.vars['spline_coeffs'][phase]
            
            for tau in self.taus_to_check:
                L_dot = self.evaluate_angular_momentum_derivative(tau)
                prog.AddQuadraticCost(w * Tk * L_dot.dot(L_dot))


    # COST HELPERS

    def compute_nominal_stance_error(self, p_foot, base_pos, base_rot):
        """
        Helper for nominal_kinematics_cost - computes error between 
        actual and desired leg length/configuration
        """
        # Get hip position in world frame
        leg_idx = self.get_leg_index(p_foot)
        hip_offset_body = self.hip_offsets[leg_idx]
        hip_pos_world = base_pos + base_rot @ hip_offset_body
        
        # Vector from hip to foot
        leg_vector = p_foot - hip_pos_world
        
        # Nominal configuration vector (straight down at desired height)
        nominal_vector = np.array([0, 0, -self.nominal_height])
        
        return leg_vector - nominal_vector

    def compute_foot_collision_metric(self, p_i, p_j):
        """
        Helper for leg_collision_cost - computes smooth collision metric
        between two foot positions
        """
        # Distance in xy-plane
        xy_diff = p_i[:2] - p_j[:2]
        dist_xy = np.sqrt(xy_diff.dot(xy_diff))
        
        # Smooth barrier function
        min_dist = 2 * self.foot_radius  # Minimum distance between foot centers
        if dist_xy >= min_dist:
            return 0.0
        else:
            # Quadratic penalty that grows as feet get closer
            violation = min_dist - dist_xy
            return violation * violation

    def evaluate_terrain_edge_cost(self, p_foot):
        """
        Helper for edge_avoidance_cost - computes cost based on 
        terrain gradients and curvature
        """
        x, y = p_foot[:2]
        
        # Get gradients from both raw and smoothed height maps
        grad_h = self.evaluate_height_gradient(x, y)
        grad_hs1 = self.evaluate_smoothed_height_gradient(x, y)
        
        # Gradient magnitudes
        grad_cost = grad_h.dot(grad_h) + grad_hs1.dot(grad_hs1)
        
        # Could also add curvature terms here for additional edge detection
        return grad_cost

    def compute_stance_cost(self, phase_idx, tau):
        """
        Helper for base_alignment_cost - computes cost for base pose
        relative to stance feet
        """
        a_k = self.vars['spline_coeffs'][phase_idx]
        pos_B = self.evaluate_spline_position(a_k, tau)
        R_B = self.compute_rotation_matrix(pos_B[3:6])
        
        cost = 0.0
        stance_feet = self.get_stance_feet(phase_idx)
        
        for leg_idx in stance_feet:
            # Get desired hip height from smoothed terrain
            hip_offset_body = self.hip_offsets[leg_idx]
            hip_pos_world = pos_B[:3] + R_B @ hip_offset_body
            h_des = self.evaluate_smoothed_height_at_xy(hip_pos_world[0], 
                                                      hip_pos_world[1])
            
            # Error between actual and desired hip height
            error = hip_pos_world[2] - (h_des + self.nominal_height)
            cost += error * error
            
        return cost

    def compute_tracking_error(self, phase_idx, tau):
        """
        Helper for tracking_cost - computes weighted error between
        current and reference state
        """
        Tk = self.phase_durations[phase_idx]
        a_k = self.vars['spline_coeffs'][phase_idx]
        
        # Current state
        pos = self.evaluate_spline_position(a_k, tau)
        vel = self.evaluate_spline_velocity(a_k, tau, Tk)
        
        # Reference state at this time
        t = self.phase_timing[phase_idx] + tau * Tk
        pos_ref = self.reference_trajectory['pos'](t)
        vel_ref = self.reference_trajectory['vel'](t)
        
        # Compute weighted errors
        pos_error = pos - pos_ref
        vel_error = vel - vel_ref
        
        # Weight linear and angular components differently
        I3 = np.eye(3)
        error_weighted = np.concatenate([
            pos_error[:3] @ (I3/self.mass),
            pos_error[3:] @ self.inertia,
            vel_error[:3],
            vel_error[3:]
        ])
        
        return error_weighted

    def compute_momentum_smoothness(self, phase_idx, tau):
        """
        Helper for smoothness_cost - computes smoothness metric
        based on angular momentum rate of change
        """
        Tk = self.phase_durations[phase_idx]
        a_k = self.vars['spline_coeffs'][phase_idx]
        
        # Get current state
        pos = self.evaluate_spline_position(a_k, tau)
        vel = self.evaluate_spline_velocity(a_k, tau, Tk)
        acc = self.evaluate_spline_acceleration(a_k, tau, Tk)
        
        # Compute angular momentum derivative
        R = self.compute_rotation_matrix(pos[3:6])
        I_world = R @ self.inertia @ R.T
        omega = vel[3:]
        
        # L_dot = I * omega_dot + omega × (I * omega)
        L_dot = I_world @ acc[3:] + np.cross(omega, I_world @ omega)
        
        return L_dot.dot(L_dot)

    def get_leg_index(self, p_foot):
        """Helper to identify which leg a foot position belongs to"""
        # Find closest nominal foot position to identify leg
        dists = []
        for i, hip_offset in enumerate(self.hip_offsets):
            d = np.linalg.norm(p_foot[:2] - hip_offset[:2])
            dists.append(d)
        return np.argmin(dists)







    # GENERAL HELPERS

    def get_phase_at_time(self, t, gait_pattern):
        """Get which phase index and normalized time within phase"""
        phase_times = gait_pattern['phase_timing']
        for i in range(len(phase_times)-1):
            if phase_times[i] <= t <= phase_times[i+1]:
                # Normalize time to [0,1] within phase
                tau = (t - phase_times[i]) / (phase_times[i+1] - phase_times[i])
                return i, tau
        raise ValueError(f"Time {t} not within gait pattern")

    def evaluate_spline(self, t, gait_pattern):
        """Evaluate spline at time t"""
        phase_idx, tau = self.get_phase_at_time(t, gait_pattern)
        coeffs = self.vars['spline_coeffs'][phase_idx]
        
        # Evaluate quintic polynomial
        pos = np.zeros(self.base_dims)
        vel = np.zeros(self.base_dims)
        acc = np.zeros(self.base_dims)
        
        for i in range(self.spline_order):
            pos += coeffs[:,i] * tau**i
            if i > 0:
                vel += i * coeffs[:,i] * tau**(i-1)
            if i > 1:
                acc += i * (i-1) * coeffs[:,i] * tau**(i-2)
        
    def determinant(self, a, b, c):
        """Compute scalar triple product a·(b×c)"""
        return a.dot(np.cross(b, c))
        
    def evaluate_spline_position(self, coeffs, tau):
        """Evaluate spline position at normalized time tau"""
        pos = 0
        for i in range(self.spline_order):
            pos += coeffs[:,i] * tau**i
        return pos  # Return full 6D pose vector
        
    def evaluate_spline_velocity(self, coeffs, tau, T):
        """Evaluate spline velocity at normalized time tau"""
        vel = np.zeros(self.base_dims)
        for i in range(1, self.spline_order):
            vel += i * coeffs[:,i] * tau**(i-1) / T
        return vel

    def evaluate_spline_acceleration(self, coeffs, tau, T):
        """Evaluate spline acceleration at normalized time tau"""
        acc = 0
        for i in range(2, self.spline_order):
            acc += i * (i-1) * coeffs[:,i] * tau**(i-2) / (T*T)
        return acc[:3]  # Only return x,y,z acceleration

    def evaluate_angular_momentum_derivative(self, tau):
        """Evaluate rate of change of angular momentum at normalized time tau"""
        # For initial testing, can return zero
        # TODO: Full implementation would compute based on robot state
        return np.zeros(3)

    def compute_rotation_matrix(self, rpy):
        """Compute rotation matrix from roll-pitch-yaw angles"""
        cr, cp, cy = np.cos(rpy)
        sr, sp, sy = np.sin(rpy)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        
        return R

    def get_num_contacts(self, phase):
        """Get number of feet in contact for given phase"""
        # Store gait pattern when received in solve()
        if not hasattr(self, 'gait_pattern'):
            raise ValueError("Gait pattern not set")

        contact_states = self.gait_pattern['contact_states'][phase]
        return sum(contact_states)


    # HEIGHT MAP
    
    def set_height_maps(self, height_map, height_map_smoothed, x_min, x_max, y_min, y_max):
        """Set the height maps and create interpolators"""
        self.height_map = height_map
        self.height_map_smoothed = height_map_smoothed
        self.x_range = [x_min, x_max]
        self.y_range = [y_min, y_max]

    # TODO: FIGURE THIS OUT
    def evaluate_height_at_xy(self, x, y):
        """Evaluate terrain height at given x,y coordinates"""
        # This should interface with your height map
        # For now returning 0 as placeholder
        return 0.0 # hard code 0 <=> flat ground

    def evaluate_smoothed_height_at_xy(self, x, y):
        """Evaluate smoothed terrain height at given x,y point"""
        if self.height_map_smoothed is None:
            return 0.0
        return float(self.height_map_smoothed([x, y]))
        


    def get_stance_feet(self, phase):
        """Get indices of feet in contact for given phase"""
        if not hasattr(self, 'gait_pattern'):
            raise ValueError("Gait pattern not set")
                
        # Return indices where contact state is 1
        contact_states = self.gait_pattern['contact_states'][phase]
        return [i for i, in_contact in enumerate(contact_states) if in_contact]

    def get_contact_pairs(self, stance_feet):
        """Get all pairs of stance feet indices"""
        # Return all unique pairs of indices from stance_feet
        return [(i, j) for i in stance_feet for j in stance_feet if i < j]



    def solve(self, current_state, gait_pattern):
        """Solve the optimization problem"""
        

        # Update current state
        self.base_pose = current_state['base_pose']
        self.base_vel = current_state['base_vel']
        self.p_meas = current_state['foot_positions']
        self.gait_pattern = gait_pattern
        
        # Setup and solve optimization
        self.setup_variables(gait_pattern)
        
        # Add constraints one by one and check feasibility
        try:
            
            self.add_initial_constraints()
            
            self.add_dynamics_constraints() # gives a division by zero error :( -- FIXED
            
            self.add_kinematic_constraints()
            
            self.add_gait_constraints(gait_pattern) # some infeasible error :( -- IGNORED
            
            self.add_costs(self.prog)

            print("Solving optimization...")
            result = Solve(self.prog)
        
            if not result.is_success():
                print(f"Optimization failed with status: {result.get_solution_result()}")
                # Print infeasible constraints if available
                if hasattr(result, 'GetInfeasibleConstraints'):
                    infeasible = result.GetInfeasibleConstraints(self.prog)
                    print(f"Infeasible constraints: {infeasible}")
                
                raise RuntimeError("Optimization failed to solve!")
                
                
            num_phases = len(gait_pattern['phase_timing'])
            solution = {
                'base_traj': [result.GetSolution(self.vars['spline_coeffs'][i]) for i in range(num_phases - 1)],
                'footholds': result.GetSolution(self.vars['p']),
                'cost': result.get_optimal_cost()
            }
            
            return solution
            
        except Exception as e:
            print(f"Error during optimization: {e}")
            raise