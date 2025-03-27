# animator.py
"""
animator.py

This module encapsulates the animation routines for the Quantum DNA simulation.
It provides two classes: Animator1D and Animator2D.

Requires:
  - quantum_ops.py (the quantum evolution routines)
  - constants.py for domain extents if needed
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import quantum_ops
from constants import EV_TO_J

class Animator1D:
    """
    Animator1D: Controls the time evolution for a 1D wavefunction using a split-operator
    and optional environment/boundary conditions.
    """

    def __init__(self, psi_init, x, dx, kx, dt, steps_per_frame, ui_vars,
                 plot_objs, data_arrays, callback_after):
        """
        psi_init: initial 1D wavefunction (numpy array)
        x: spatial grid (1D array)
        dx: grid spacing
        kx: wave number array (1D)
        dt: time step
        steps_per_frame: how many time-evolution steps per "animation" call
        ui_vars: dictionary of tkinter variables, e.g.:
                 {
                   'D': DoubleVar, 'a': DoubleVar, 'x0': DoubleVar, 'C': DoubleVar,
                   'T': DoubleVar, 'R': DoubleVar,
                   'Boundary': StringVar,       # "none", "fade", "cap", "both"
                   'Environment': StringVar,    # "none", "therm", "stoch", "both"
                   'CAPalpha': DoubleVar,
                   'Inflow': DoubleVar,
                   'X_MIN_1D': DoubleVar,
                   'X_MAX_1D': DoubleVar,
                   'ConstantT': BooleanVar
                 }
        plot_objs: dictionary with relevant matplotlib objects for updating plots
                 e.g.:
                 {
                   'ax_pot': ax_pot,
                   'line_pot': line_pot,
                   'ax_psi': ax_psi,
                   'canvas_psi': canvas_psi,
                   'ax_en': ax_en,
                   'canvas_en': canvas_en,
                   'ax_prob': ax_prob,
                   'canvas_prob': canvas_prob,
                   'canvas_pot': canvas_pot  # so we can force pot redraw
                 }
        data_arrays: dictionary for storing time, energy, probabilities, plus references
                     to compute_energy and prob_left_right:
                     {
                       'time': [],
                       'energy': [],
                       'pleft': [],
                       'pright': [],
                       'compute_energy': function,
                       'prob_left_right': function
                     }
        callback_after: function for scheduling the next frame (e.g., root.after in Tk)
        """

        self.psi_init = psi_init.copy()
        self.psi = psi_init.copy()
        self.x = x
        self.dx = dx
        self.kx = kx
        self.dt = dt
        self.steps = steps_per_frame
        self.ui = ui_vars
        self.plots = plot_objs
        self.data = data_arrays
        self.callback_after = callback_after

        self.frame_counter = 0
        self.anim_running = False

    def build_potential(self):
        """
        Build the effective potential V_eff for the 1D wavefunction,
        including random thermal/radiation noise, plus an imaginary CAP if selected.
        """
        D = self.ui['D'].get()
        a_val = self.ui['a'].get()
        x0_val = self.ui['x0'].get()
        C = self.ui['C'].get()

        # base double Morse
        V_morse = quantum_ops.build_morse_1D(self.x, D, a_val, x0_val, C)

        # environment/thermal noise
        env_mode = self.ui['Environment'].get()  # "none", "therm", "stoch", "both"
        if env_mode in ["therm", "both"]:
            # small amplitude random potential
            # (the "small perturbation" you requested)
            dU_amp = 0.00001 * EV_TO_J * (self.ui['T'].get() / 300.0)
            V_morse += dU_amp * np.random.randn(len(self.x))

        # radiation noise (R > 0)
        if self.ui['R'].get() > 0.01:
            rad_amp = self.ui['R'].get() * 0.01 * EV_TO_J
            V_morse += rad_amp * np.random.randn(len(self.x))

        # boundary/cap
        boundary_mode = self.ui['Boundary'].get()  # "none", "fade", "cap", "both"
        if boundary_mode in ["cap", "both"]:
            cap_alpha = self.ui['CAPalpha'].get()
            x_min = self.ui['X_MIN_1D'].get()
            x_max = self.ui['X_MAX_1D'].get()
            V_cap = quantum_ops.cap_function_1D(self.x, x_min, x_max,
                                                cap_width=0.1e-10,  # narrower by default
                                                alpha=cap_alpha)
            # final potential: real part + i*(-cap)
            V_eff = V_morse + 1j*(-V_cap)
        else:
            V_eff = V_morse

        return V_eff

    def animate(self):
        """ One animation step: do multiple time-evolution steps, then update plots. """
        if not self.anim_running:
            return

        V_eff = self.build_potential()
        gamma = 1e14 * (self.ui['T'].get() / 300.0)  # Lindblad scale

        for _ in range(self.steps):
            self.psi = quantum_ops.split_operator_step_1D(self.psi, V_eff, self.dt, self.kx)
            self.psi = quantum_ops.apply_lindblad_1D(self.psi, self.dt, gamma)

            boundary_mode = self.ui['Boundary'].get()
            if boundary_mode in ["fade", "both"]:
                self.psi = quantum_ops.fade_edges_1D(self.psi)

            env_mode = self.ui['Environment'].get()
            if env_mode in ["stoch", "both"]:
                inflow = self.ui['Inflow'].get()
                self.psi = quantum_ops.stochastic_inflow_1D(self.psi, self.dx, inflow, self.dt)

            # Optionally keep temperature roughly constant:
            if self.ui['ConstantT'].get():
                self.psi = quantum_ops.apply_constant_temperature_1D(
                    self.psi, np.real(V_eff), self.dt,
                    self.ui['T'].get(),
                    self.data['compute_energy'],
                    self.x, self.dx
                )

        # Update Potential Plot
        self.plots['line_pot'].set_ydata(np.real(V_eff)/EV_TO_J)
        self.plots['ax_pot'].relim()
        self.plots['ax_pot'].autoscale_view()

        # Force potential canvas to refresh, so it updates in real time
        if 'canvas_pot' in self.plots:
            self.plots['canvas_pot'].draw_idle()

        # Update wavefunction plot every ~5 frames
        if self.frame_counter % 5 == 0:
            psi_sq = np.abs(self.psi)**2
            psi_sq_smooth = gaussian_filter1d(psi_sq, sigma=1)
            ax_psi = self.plots['ax_psi']
            ax_psi.clear()
            ax_psi.plot(self.x*1e10, psi_sq_smooth, '-')
            ax_psi.set_title("Wavefunction")
            self.plots['canvas_psi'].draw_idle()

        # Update energy and probabilities every ~2 frames
        if self.frame_counter % 2 == 0:
            E_val = self.data['compute_energy'](self.psi, np.real(V_eff))
            self.data['time'].append(self.frame_counter * self.dt)
            self.data['energy'].append(E_val)

            pleft, pright = self.data['prob_left_right'](self.psi)
            self.data['pleft'].append(pleft)
            self.data['pright'].append(pright)

            ax_en = self.plots['ax_en']
            ax_en.clear()
            ax_en.plot(np.array(self.data['time'])*1e15,
                       np.array(self.data['energy'])/EV_TO_J, '-')
            ax_en.set_title("Energy vs Time")
            ax_en.set_xlabel("Time (fs)")
            ax_en.set_ylabel("Energy (eV)")
            self.plots['canvas_en'].draw_idle()

            ax_prob = self.plots['ax_prob']
            ax_prob.clear()
            ax_prob.plot(np.array(self.data['time'])*1e15, self.data['pleft'], '-', label="P_left")
            ax_prob.plot(np.array(self.data['time'])*1e15, self.data['pright'], '-', label="P_right")
            ax_prob.set_title("Tunneling Probability")
            ax_prob.legend()
            self.plots['canvas_prob'].draw_idle()

        self.frame_counter += 1
        self.callback_after(50, self.animate)  # schedule next frame

    def start(self):
        if not self.anim_running:
            self.anim_running = True
            self.frame_counter = 0
            self.animate()

    def stop(self):
        self.anim_running = False


class Animator2D:
    """
    Animator2D: Controls the time evolution for a 2D wavefunction using split-operator
    and optional environment/boundary conditions.
    """

    def __init__(self, psi_init, X, dx, kx, ky, dt, steps_per_frame, ui_vars,
                 plot_objs, data_arrays, callback_after):
        """
        psi_init: initial 2D wavefunction (numpy array)
        X: (X1, X2) meshgrids
        dx: grid spacing (assume same in x and y)
        kx, ky: 2D wave number arrays from np.meshgrid
        dt: time step
        steps_per_frame: number of evolution steps per frame
        ui_vars: dict of tkinter variables for boundary, environment, etc.
        plot_objs: e.g. {'ax_dyn': ax_2D_dynamic, 'canvas_dyn': canvas_2D_dynamic}
        data_arrays: dictionary for storing any needed data
        callback_after: function to schedule next step (e.g. root.after)
        """
        self.psi_init = psi_init.copy()
        self.psi = psi_init.copy()
        self.X = X  # (X1, X2)
        self.dx = dx
        self.kx = kx
        self.ky = ky
        self.dt = dt
        self.steps = steps_per_frame
        self.ui = ui_vars
        self.plots = plot_objs
        self.data = data_arrays
        self.callback_after = callback_after

        self.frame_counter = 0
        self.anim_running = False

    def build_potential(self):
        """
        Construct the 2D potential with Morse wells, noise, and an imaginary CAP if selected.
        """
        D = self.ui['D'].get()
        a_val = self.ui['a'].get()
        x0_val = self.ui['x0'].get()
        C = self.ui['C'].get()

        # base double Morse (2D)
        V_morse = quantum_ops.build_morse_2D(self.X[0], self.X[1], D, a_val, x0_val, C)

        # environment
        env_mode = self.ui['Environment'].get()
        if env_mode in ["therm", "both"]:
            # small amplitude random potential
            dU_amp = 0.00001 * EV_TO_J * (self.ui['T'].get() / 300.0)
            V_morse += dU_amp * np.random.randn(*self.X[0].shape)

        if self.ui['R'].get() > 0.01:
            rad_amp = self.ui['R'].get() * 0.01 * EV_TO_J
            V_morse += rad_amp * np.random.randn(*self.X[0].shape)

        # boundary/cap
        boundary_mode = self.ui['Boundary'].get()
        if boundary_mode in ["cap", "both"]:
            cap_alpha = self.ui['CAPalpha'].get()
            xmin = self.ui['X_MIN_2D'].get()
            xmax = self.ui['X_MAX_2D'].get()
            Vcap_2D = quantum_ops.cap_function_2D(
                self.X[0], self.X[1], xmin, xmax,
                cap_width=0.1e-10,  # narrower
                alpha=cap_alpha
            )
            V_eff = V_morse + 1j*(-Vcap_2D)
        else:
            V_eff = V_morse

        return V_eff

    def animate(self):
        if not self.anim_running:
            return

        V_eff = self.build_potential()
        gamma = 1e14 * (self.ui['T'].get() / 300.0)

        for _ in range(self.steps):
            self.psi = quantum_ops.split_operator_step_2D(self.psi, V_eff, self.dt, self.kx, self.ky)
            self.psi = quantum_ops.apply_lindblad_2D(self.psi, self.dt, gamma)

            boundary_mode = self.ui['Boundary'].get()
            if boundary_mode in ["fade", "both"]:
                # quick fade at the edges
                self.psi[:2, :] *= 0.99
                self.psi[-2:, :] *= 0.99
                self.psi[:, :2] *= 0.99
                self.psi[:, -2:] *= 0.99

            env_mode = self.ui['Environment'].get()
            if env_mode in ["stoch", "both"]:
                inflow = self.ui['Inflow'].get()
                self.psi = quantum_ops.stochastic_inflow_2D(self.psi, self.dx, inflow, self.dt)

        # Update dynamic 2D plot
        self.plots['ax_dyn'].clear()
        psi_sq = np.abs(self.psi)**2
        psi_sq_smooth = gaussian_filter(psi_sq, sigma=1)
        xmin = self.ui['X_MIN_2D'].get() * 1e10
        xmax = self.ui['X_MAX_2D'].get() * 1e10

        self.plots['ax_dyn'].imshow(psi_sq_smooth,
                                    extent=[xmin, xmax, xmin, xmax],
                                    origin="lower", cmap="magma", aspect='auto')
        self.plots['ax_dyn'].set_title("Dynamic Wavefunction (2D)")
        self.plots['ax_dyn'].set_xlabel("x1 (Å)")
        self.plots['ax_dyn'].set_ylabel("x2 (Å)")

        self.plots['canvas_dyn'].draw_idle()

        self.frame_counter += 1
        self.callback_after(50, self.animate)

    def start(self):
        if not self.anim_running:
            self.anim_running = True
            self.frame_counter = 0
            self.animate()

    def stop(self):
        self.anim_running = False
