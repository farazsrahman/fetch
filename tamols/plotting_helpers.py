import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pydrake.symbolic import Expression, Evaluate
from tamols import TAMOLSState
from helpers import evaluate_spline_positions, evaluate_spline_velocity, evaluate_spline_position


def plot_optimal_solutions_interactive(tmls: TAMOLSState):
    optimal_footsteps = tmls.optimal_footsteps
    optimal_spline_coeffs = tmls.optimal_spline_coeffs
    num_phases = tmls.num_phases
    
    # Create figure
    fig = go.Figure()
    
    # Plot initial foot positions (p_meas)
    for i in range(tmls.p_meas.shape[0]):
        fig.add_trace(go.Scatter3d(
            x=[tmls.p_meas[i, 0]],
            y=[tmls.p_meas[i, 1]],
            z=[tmls.p_meas[i, 2]],
            mode='markers',
            name=f'p_meas {i+1}',
            marker=dict(size=8, color='black')
        ))
    
    # Colors for alternating steps
    colors = ['red', 'green', 'green', 'red']
    
    # Plot optimal footsteps
    for i in range(optimal_footsteps.shape[0]):
        fig.add_trace(go.Scatter3d(
            x=[optimal_footsteps[i, 0]],
            y=[optimal_footsteps[i, 1]],
            z=[optimal_footsteps[i, 2]],
            mode='markers+text',
            name=f'Footstep {i+1}',
            marker=dict(size=8, color=colors[i % len(colors)]),
            text=[f'Footstep {i+1}'],
            textposition='top center'
        ))
    
    # Plot splines for each phase
    for i in range(num_phases):
        tau_values = np.linspace(0, 1, 100)
        spline_points = evaluate_spline_positions(tmls, optimal_spline_coeffs[i], tau_values)
        
        fig.add_trace(go.Scatter3d(
            x=spline_points[:, 0],
            y=spline_points[:, 1],
            z=spline_points[:, 2],
            mode='lines',
            name=f'Spline Phase {i+1}',
            line=dict(color='blue', width=4),  # Make spline lines thick and blue
        ))
        
        # Add end point markers without text
        fig.add_trace(go.Scatter3d(
            x=[spline_points[-1, 0]],
            y=[spline_points[-1, 1]],
            z=[spline_points[-1, 2]],
            mode='markers',
            name=f'End {i+1}',
            marker=dict(size=6, color=colors[i % len(colors)]),
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title='Optimal Base Pose and Footsteps',
        scene=dict(
            xaxis=dict(range=[-1, 1], title='X'),
            yaxis=dict(range=[-1, 1], title='Y'),
            zaxis=dict(range=[0, 1], title='Z'),
            aspectmode='cube'  # This ensures equal scaling
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        ),
        margin=dict(r=250),  # Add right margin for legend
        showlegend=True
    )
    
    # Add grid lines
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='gray'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='gray'
            ),
            zaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='gray'
            )
        )
    )
    # Save as HTML file for interactive viewing
    fig.write_html('out/interactive_optimal_base_pose_and_footsteps.html')

def plot_optimal_solutions(tmls: TAMOLSState):
    optimal_footsteps = tmls.optimal_footsteps
    optimal_spline_coeffs = tmls.optimal_spline_coeffs
    num_phases = tmls.num_phases

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot initial foot positions (p_meas) in black
    for i in range(tmls.p_meas.shape[0]):
        ax.scatter(tmls.p_meas[i, 0], tmls.p_meas[i, 1], tmls.p_meas[i, 2], label=f'Initial Footstep {i+1}', color='k')

    colors = ['r', 'g', 'g', 'r']
    for i in range(optimal_footsteps.shape[0]):
        ax.scatter(optimal_footsteps[i, 0], optimal_footsteps[i, 1], optimal_footsteps[i, 2], label=f'Footstep {i+1}', color=colors[i % len(colors)])

    for i in range(num_phases):
        T = tmls.phase_durations[i]
        tau_values = np.linspace(0, 1, 100)

        spline_points = evaluate_spline_positions(tmls, optimal_spline_coeffs[i], tau_values)
        ax.plot(spline_points[:, 0], spline_points[:, 1], spline_points[:, 2], color='blue', linewidth=2)  # Make spline lines thick and blue
        
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Optimal Base Pose and Footsteps')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    
    plt.savefig('out/optimal_base_pose_and_footsteps.png')


def save_optimal_solutions(tmls: TAMOLSState, filepath='out/optimal_solution.txt'):
    optimal_footsteps = tmls.optimal_footsteps
    optimal_spline_coeffs = tmls.optimal_spline_coeffs
    num_phases = tmls.num_phases

    with open(filepath, 'w') as f:
        # SOLUTION
        f.write("Optimal Footsteps:\n")
        for i in range(optimal_footsteps.shape[0]):
            f.write(f"Footstep {i+1}: {optimal_footsteps[i, 0]}, {optimal_footsteps[i, 1]}, {optimal_footsteps[i, 2]}\n")
        
        f.write("\nOptimal Spline Coefficients:\n")
        for i in range(num_phases):
            f.write(f"Spline Phase {i+1} Coefficients:\n")
            np.savetxt(f, optimal_spline_coeffs[i], fmt='%.6f')
            f.write("\n")

        f.write("\nObjective Function Optimal Value:\n")
        optimal_value = tmls.result.get_optimal_cost()  # Assuming this method exists
        f.write(f"Optimal Value: {optimal_value:.6f}\n")

       
        # VELOCITY
        f.write("\nReference Velocity:\n")
        ref_vel = tmls.ref_vel  # Assuming this attribute exists
        f.write(f"Reference Velocity: {ref_vel}\n")

        # f.write("\nVelocity at Time Steps Tau:\n")
        # for i in range(num_phases):
        #     T_k = tmls.phase_durations[i]
        #     for tau in np.linspace(0, T_k, tmls.tau_sampling_rate+1)[:tmls.tau_sampling_rate]:
        #         velocity_at_tau = evaluate_spline_velocity(tmls, optimal_spline_coeffs[i], tau, 1)  # Assume T_k = 1 forall k
        #         f.write(f"Velocity at tau={tau:.2f}: {np.array2string(velocity_at_tau, formatter={'float_kind':lambda x: f'{x:.4f}'})}\n")

        # FOOT DISTANCES AND COSTS
        f.write("\nFoot Distances and Associated Costs:\n")
        num_legs = tmls.num_legs

        for i in range(num_legs):
            for j in range(i + 1, num_legs):
                dist = np.linalg.norm(optimal_footsteps[i] - optimal_footsteps[j])
                f.write(f"Foot {i+1} to {j+1}: {dist:.6f}\n")

        

        f.write("\nDistance between leg index and hip location for kinematic constraints:\n")
        for leg_idx in range(tmls.num_legs):
            for phase_idx, at_des_pos in enumerate(tmls.gait_pattern['at_des_position']):
                if at_des_pos[leg_idx]:
                    T_k = tmls.phase_durations[phase_idx]
                    for tau in np.linspace(0, T_k, tmls.tau_sampling_rate+1)[:tmls.tau_sampling_rate]:
                        spline_coeffs_solution = tmls.result.GetSolution(tmls.spline_coeffs[phase_idx])
                        p_solution = tmls.result.GetSolution(tmls.p[leg_idx])
                        
                        spline_pos = evaluate_spline_position(tmls, spline_coeffs_solution, tau)[0:3]
                        hip_location = spline_pos + tmls.hip_offsets[leg_idx]
                        distance = np.linalg.norm(p_solution - hip_location)
                        f.write(f"Leg {leg_idx+1}, Phase {phase_idx+1}, Tau {tau:.2f}: Distance = {distance:.6f}, Actual Foot Location = {p_solution}, Hip Location = {spline_pos + tmls.hip_offsets[leg_idx]}\n")
