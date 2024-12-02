{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The autoreload extension is already loaded. To reload it, use:\n",
      "  %reload_ext autoreload\n"
     ]
    }
   ],
   "source": [
    "%load_ext autoreload\n",
    "%autoreload 2\n",
    "\n",
    "from utils import TAMOLS\n",
    "\n",
    "model = TAMOLS()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "def create_test_gait_pattern():\n",
    "    \"\"\"Create a simple trotting gait pattern\"\"\"\n",
    "    # Simple two-phase trot (diagonal legs in contact)\n",
    "    return {\n",
    "        'phase_timing': [0.0, 0.25, 0.5],  # Two phases of 0.25s each\n",
    "        'contact_states': [\n",
    "            [1, 0, 0, 1],  # Phase 0: FL and BR in contact\n",
    "            [0, 1, 1, 0],  # Phase 1: FR and BL in contact\n",
    "        ]\n",
    "    }\n",
    "\n",
    "def create_initial_state():\n",
    "    \"\"\"Create a simple initial state\"\"\"\n",
    "    return {\n",
    "        'base_pose': np.array([0., 0., 0.5, 0., 0., 0.]),  # x,y,z,roll,pitch,yaw\n",
    "        'base_vel': np.zeros(6),  # zero initial velocity\n",
    "        'foot_positions': np.array([\n",
    "            [ 0.2,  0.15, 0.0],  # FL\n",
    "            [ 0.2, -0.15, 0.0],  # FR\n",
    "            [-0.2,  0.15, 0.0],  # BL\n",
    "            [-0.2, -0.15, 0.0],  # BR\n",
    "        ])\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Solving optimization...\n",
      "Adding initial constraints...\n",
      "Optimization succeeded!\n",
      "Final cost: 0.0\n",
      "\n",
      "Base trajectory first phase coefficients:\n",
      "[[-3.04696766e-18  8.96604206e-34 -1.08142987e-17  1.28252107e-18\n",
      "   3.81271105e-18]\n",
      " [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00\n",
      "   0.00000000e+00]\n",
      " [ 5.00000000e-01 -2.25514052e-17 -1.53846154e-01 -3.84615385e-02\n",
      "   7.69230769e-02]\n",
      " [ 4.61110735e-33  1.27648143e-33  4.72397007e-17 -4.31708884e-18\n",
      "  -1.71690448e-17]\n",
      " [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00\n",
      "   0.00000000e+00]\n",
      " [-2.90094631e-18 -1.89802087e-17  9.34123999e-18 -2.71677498e-18\n",
      "   2.53069450e-18]]\n",
      "\n",
      "Optimized footholds:\n",
      "[[ 0.2   0.15  0.  ]\n",
      " [ 0.2  -0.15  0.  ]\n",
      " [-0.2   0.15  0.  ]\n",
      " [-0.2  -0.15  0.  ]]\n"
     ]
    }
   ],
   "source": [
    "# Create planner instance\n",
    "planner = TAMOLS(dt=0.01, horizon_steps=50)\n",
    "\n",
    "# Get test data\n",
    "gait_pattern = create_test_gait_pattern()\n",
    "initial_state = create_initial_state()\n",
    "\n",
    "try:\n",
    "    # Solve feasibility problem\n",
    "    solution = planner.solve(initial_state, gait_pattern)\n",
    "    print(\"Optimization succeeded!\")\n",
    "    print(f\"Final cost: {solution['cost']}\")\n",
    "    \n",
    "    # Print some solution details\n",
    "    print(\"\\nBase trajectory first phase coefficients:\")\n",
    "    print(solution['base_traj'][0])\n",
    "    print(\"\\nOptimized footholds:\")\n",
    "    print(solution['footholds'])\n",
    "    \n",
    "except RuntimeError as e:\n",
    "    print(f\"Optimization failed: {e}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "meam",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.20"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
