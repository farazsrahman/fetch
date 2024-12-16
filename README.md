Repo for MEAM 5170 project using hierarchical model-based motion planner and RL whole body controller for the Unitree Go2

A few examples of motion plans can are shown below:
Going up stairs:

![ezgif-7-1fc539f22a](https://github.com/user-attachments/assets/86f61f9d-b23f-4f54-acb0-fc07176b5d85)

Navigating floor with holes:

<img width="505" alt="Screenshot 2024-12-09 at 2 20 26 AM" src="https://github.com/user-attachments/assets/f10dfdce-82a4-4f2c-9b12-f328b5d15734" />


To run this install `pydrake` and `plotly` (or whatever it asks for you to install), and then run python `/tamols/test.py` from the root dir/. If a feasible solution is found to the test problem, this will output an html file that is an interactive 3d view of the motion plan.
