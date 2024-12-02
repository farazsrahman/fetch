#pragma once

#include <Eigen/Dense>
#include <drake/solvers/mathematical_program.h>
#include <vector>
#include <string>
#include <map>

namespace tamols {

class TAMOLSState {
public:
    TAMOLSState();
    void setupVariables();

    // Time and horizon parameters
    double dt;
    int horizon_steps;

    // Height map data
    Eigen::MatrixXd height_map;
    Eigen::MatrixXd height_map_smoothed;

    // Robot configuration
    int num_legs;
    int spline_order;
    int base_dims;

    // Physical properties
    double mass;
    double mu;
    Eigen::Matrix3d inertia;

    // Kinematic constraints
    Eigen::MatrixXd hip_offsets;
    double l_min;
    double l_max;

    // State measurements
    Eigen::VectorXd p_meas;
    Eigen::VectorXd base_pose;
    Eigen::VectorXd base_vel;

    // Optimization parameters
    std::vector<double> taus_to_check;
    std::map<std::string, double> weights;

    // Optimization program
    drake::solvers::MathematicalProgram prog;
    
    // Spline coefficients for trajectory
    std::vector<
        Eigen::Matrix<
            drake::symbolic::Variable,
            Eigen::Dynamic,
            Eigen::Dynamic
        >
    > spline_coeffs;

    // Optimization variables
    Eigen::Matrix<
        drake::symbolic::Variable,
        Eigen::Dynamic,
        Eigen::Dynamic
    > p;
    Eigen::Matrix<
        drake::symbolic::Variable,
        Eigen::Dynamic,
        1
    > epsilon;

    // Height and distance parameters
    double nominal_height;
    double foot_radius;
    double min_foot_distance;
    double desired_height;

    // Trajectory and gait pattern
    Eigen::MatrixXd reference_trajectory;
    std::map<std::string, std::vector<double>> gait_pattern;

private:
    std::vector<double> phase_durations;
};

} // namespace tamols
