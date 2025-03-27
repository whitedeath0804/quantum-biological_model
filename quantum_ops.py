# quantum_ops.py
"""
This module implements quantum evolution routines including the split-operator method,
Morse potential building, boundary condition operators (CAP, fade), Lindblad-type operators,
and stochastic inflow functions for both 1D and 2D simulations.

Requires:
  - constants.py in the same directory
"""

import numpy as np
from constants import HBAR, EV_TO_J, M_PROTON

######################################
# 1D Quantum Operations
######################################

def build_morse_1D(x, D_eV, a_val, x0_val, C_eV):
    """
    Build a symmetrical 'double Morse' potential in 1D around ±x0_val.
    """
    D = D_eV * EV_TO_J
    C = C_eV * EV_TO_J
    V = (
        D * (1 - np.exp(-a_val*(x - x0_val)))**2
      + D * (1 - np.exp(-a_val*(x + x0_val)))**2
      - C
    )
    return V

def split_operator_step_1D(psi, V, dt, kx):
    """
    Advance the 1D wavefunction psi one time step using the split-operator method.
    """
    T_k = (HBAR**2)*(kx**2)/(2*M_PROTON)
    phase_k = np.exp(-1j * T_k * dt/(2*HBAR))

    psi_k = np.fft.fft(psi)
    psi_k *= phase_k
    psi_half = np.fft.ifft(psi_k)

    phase_x = np.exp(-1j * V * dt/HBAR)
    psi_half *= phase_x

    psi_k = np.fft.fft(psi_half)
    psi_k *= phase_k
    psi_new = np.fft.ifft(psi_k)

    return psi_new

def apply_lindblad_1D(psi, dt, gamma):
    """
    Apply a simple Lindblad-type operator to the 1D wavefunction for decoherence/damping.
    """
    if gamma <= 0:
        return psi
    damping = np.exp(-0.5 * gamma * dt)
    phase = np.exp(1j * np.random.normal(0, gamma*dt, size=psi.shape))
    return psi * damping * phase

def fade_edges_1D(psi, fade_factor=0.99, n_edges=3):
    """
    Fade out the edges to reduce reflection at the boundary.
    """
    psi[:n_edges] *= fade_factor
    psi[-n_edges:] *= fade_factor
    return psi

def cap_function_1D(x, x_min, x_max, cap_width=0.1e-10, alpha=5.0e9):
    """
    Returns a real array for the CAP region at the edges.
    By default, narrower (0.1e-10) so it's less aggressive.
    """
    Vcap = np.zeros_like(x, dtype=np.float64)
    left_mask = (x < (x_min + cap_width))
    right_mask = (x > (x_max - cap_width))

    if np.any(left_mask):
        left_dist = (x[left_mask] - (x_min + cap_width)) / cap_width
        Vcap[left_mask] = alpha * left_dist**2

    if np.any(right_mask):
        right_dist = (x[right_mask] - (x_max - cap_width)) / cap_width
        Vcap[right_mask] = alpha * right_dist**2

    return Vcap

def stochastic_inflow_1D(psi, dx, inflow_rate, dt):
    """
    Inject random noise amplitude into the 1D wavefunction at a rate = inflow_rate.
    """
    if inflow_rate <= 0:
        return psi
    injection = inflow_rate * dt * (np.random.randn(len(psi)) + 1j*np.random.randn(len(psi)))
    new_psi = psi + injection

    norm = np.sqrt(np.sum(np.abs(new_psi)**2) * dx)
    if norm > 1e-30:
        new_psi /= norm
    return new_psi

def apply_constant_temperature_1D(psi, V, dt, T, measure_energy_func, x, dx):
    """
    A crude feedback-based method to partially maintain 'constant temperature.'
    """
    k_B = 1.380649e-23
    E_now = measure_energy_func(psi, V)
    E_target = k_B * T

    if E_now < 0.9 * E_target:
        injection_scale = 0.001 * (E_target - E_now)
        if injection_scale > 0:
            injection = injection_scale * (np.random.randn(len(psi)) + 1j*np.random.randn(len(psi)))
            psi_new = psi + injection
            norm = np.sqrt(np.sum(np.abs(psi_new)**2) * dx)
            if norm > 1e-30:
                psi = psi_new / norm
    elif E_now > 1.1 * E_target:
        factor = 1.0 - 0.00005 * (E_now - E_target) / E_now
        factor = max(0.8, factor)
        psi *= factor

    return psi

######################################
# 2D Quantum Operations
######################################

def build_morse_2D(X1, X2, D_eV, a_val, x0_val, C_eV):
    """
    Build a 'double Morse' potential in 2D with an optional coupling term.
    """
    D = D_eV * EV_TO_J
    C = C_eV * EV_TO_J
    term1 = D * (1 - np.exp(-a_val*(X1 - x0_val)))**2
    term2 = D * (1 - np.exp(-a_val*(X1 + x0_val)))**2
    term3 = D * (1 - np.exp(-a_val*(X2 - x0_val)))**2
    term4 = D * (1 - np.exp(-a_val*(X2 + x0_val)))**2
    coupling = 0.05 * D * np.abs(X1 - X2)
    V = term1 + term2 + term3 + term4 + coupling - C
    return V

def split_operator_step_2D(psi, V, dt, kx, ky):
    """
    Advance the 2D wavefunction psi one time step using the split-operator method.
    """
    T_k = (HBAR**2)*(kx**2 + ky**2)/(2*M_PROTON)
    phase_k = np.exp(-1j * T_k * dt/(2*HBAR))

    psi_k = np.fft.fft2(psi)
    psi_k *= phase_k
    psi_half = np.fft.ifft2(psi_k)

    phase_x = np.exp(-1j * V * dt/HBAR)
    psi_half *= phase_x

    psi_k = np.fft.fft2(psi_half)
    psi_k *= phase_k
    psi_new = np.fft.ifft2(psi_k)

    return psi_new

def cap_function_2D(X, Y, x_min, x_max, cap_width=0.1e-10, alpha=5.0e9):
    """
    2D Complex Absorbing Potential. By default, narrower (0.1e-10).
    """
    Vcap = np.zeros_like(X, dtype=np.float64)

    left_mask = X < (x_min + cap_width)
    right_mask = X > (x_max - cap_width)
    top_mask = Y > (x_max - cap_width)
    bot_mask = Y < (x_min + cap_width)

    if np.any(left_mask):
        left_dist = (X[left_mask] - (x_min + cap_width)) / cap_width
        Vcap[left_mask] = alpha * left_dist**2

    if np.any(right_mask):
        right_dist = (X[right_mask] - (x_max - cap_width)) / cap_width
        Vcap[right_mask] = alpha * right_dist**2

    if np.any(top_mask):
        top_dist = (Y[top_mask] - (x_max - cap_width)) / cap_width
        Vcap[top_mask] = alpha * top_dist**2

    if np.any(bot_mask):
        bot_dist = (Y[bot_mask] - (x_min + cap_width)) / cap_width
        Vcap[bot_mask] = alpha * bot_dist**2

    return Vcap

def apply_lindblad_2D(psi, dt, gamma):
    """
    Apply a Lindblad-type damping operator to 2D wavefunction.
    """
    if gamma <= 0:
        return psi
    damping = np.exp(-0.5*gamma*dt)
    phase = np.exp(1j*np.random.normal(0, gamma*dt, size=psi.shape))
    return psi * damping * phase

def stochastic_inflow_2D(psi, dx, inflow_rate, dt):
    """
    Inject random noise amplitude into a 2D wavefunction.
    """
    if inflow_rate <= 0:
        return psi
    injection = inflow_rate * dt * (np.random.randn(*psi.shape) + 1j*np.random.randn(*psi.shape))
    new_psi = psi + injection

    area = dx*dx
    norm = np.sqrt(np.sum(np.abs(new_psi)**2) * area)
    if norm > 1e-30:
        new_psi /= norm
    return new_psi
