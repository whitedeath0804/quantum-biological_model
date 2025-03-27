# constants.py
"""
This module defines the physical constants and default simulation parameters.
"""

# Physical constants
HBAR = 1.054571817e-34         # Planck's constant / 2π in J·s
EV_TO_J = 1.602176634e-19      # 1 eV in Joules
M_PROTON = 1.6735575e-27       # Proton mass in kg

# Default domain settings for 1D simulation
X_MIN_1D = -1.5e-10            # Minimum x in meters
X_MAX_1D = 1.5e-10             # Maximum x in meters
N_1D = 1024                    # Number of grid points in 1D

# Default domain settings for 2D simulation
X_MIN_2D = -1.5e-10            # Minimum x (and y) in meters
X_MAX_2D = 1.5e-10             # Maximum x (and y) in meters
N_2D = 128                     # Number of grid points per dimension in 2D

# Default simulation time parameters
DEFAULT_DT = 0.5e-16           # Default time step (s)
DEFAULT_STEPS_PER_FRAME = 20   # Default number of time steps per animation frame
