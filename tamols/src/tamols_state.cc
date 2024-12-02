#include "tamols/tamols_state.h"

namespace tamols {

TAMOLSState::TAMOLSState()
    : dt(0.01),
      horizon_steps(100),
      num_legs(4),
      spline_order(5),
      base_dims(6),
      mass(1.0),
      mu(0.7),
      inertia((Eigen::Matrix3d() << 0.07, 0, 0, 0, 0.26, 0, 0, 0, 0.242).finished()),
      hip_offsets((Eigen::MatrixXd(4, 3) << 0.2, 0.15, 0, 0.2, -0.15, 0, -0.2, 0.15, 0, -0.2, -0.15, 0).finished()),
      l_min(0.2),
      l_max(0.5),
      nominal_height(0.4),
      foot_radius(0.02),
      min_foot_distance(0.1),
      desired_height(0.4),
      taus_to_check({0.4, 0.9}),
      weights({
          {"robustness_margin", 1.0},
          {"footholds_on_ground", 1.0},
          {"leg_collision_avoidance", 1.0},
          {"nominal_kinematics", 1.0},
          {"base_alignment", 1.0},
          {"edge_avoidance", 1.0},
          {"previous_solution_tracking", 1.0},
          {"reference_trajectory_tracking", 1.0},
          {"smoothness", 1.0}
      })
{
    // Initialize other member variables if needed
}

void TAMOLSState::setupVariables() {
    // Use the existing member variable 'prog' instead of declaring a new one

    const auto& phase_times = gait_pattern["phase_timing"];
    int num_phases = phase_times.size() - 1;
    phase_durations.resize(num_phases);

    for (int i = 0; i < num_phases; ++i) {
        phase_durations[i] = phase_times[i + 1] - phase_times[i];
    }

    spline_coeffs.resize(num_phases);
    for (int i = 0; i < num_phases; ++i) {
        // Ensure spline_coeffs is of the correct type to hold symbolic variables
        spline_coeffs[i] = prog.NewContinuousVariables(base_dims, spline_order, "a_" + std::to_string(i));
    }

    p = prog.NewContinuousVariables(num_legs, 3, "p");
    epsilon = prog.NewContinuousVariables(num_phases, "epsilon");
}

} // namespace tamols
