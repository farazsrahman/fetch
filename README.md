Repo for MEAM 5170 project using hierarchical model-based motion planner and RL whole body controller for the Unitree Go2

A few examples of motion plans can are shown below:
Going up stairs:

![ezgif-7-1fc539f22a](https://github.com/user-attachments/assets/86f61f9d-b23f-4f54-acb0-fc07176b5d85)

Navigating floor with holes:

<img width="505" alt="Screenshot 2024-12-09 at 2 20 26 AM" src="https://github.com/user-attachments/assets/f10dfdce-82a4-4f2c-9b12-f328b5d15734" />



Look at this document for notes on which of the costs/constraints have been implemented / are wip
https://docs.google.com/document/d/1sHdM0nYUNMaIfw7tmxlh4CA-Nb2IrAO7sX3AtRb7L1k/edit?usp=sharing


As of now (Tues Dec 3rd 4:22pm) I think that add_initial_constraints and add_dynamics_constraints are good. The next thing to do would be to add a cost for tracking a reference tracectory if we want to implement the costs/contraints in order of most to least important.

To run this install `pydrake` and `plotly` (or whatever it asks for you to install), and then run python `/tamols/test.py` from the root dir/. If a feasible solution is found to the test problem, this will output an html file that is an interactive 3d view of the motion plan.
