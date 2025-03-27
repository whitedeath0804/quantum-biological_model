#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Quantum DNA Simulation
--------------------------------
Features:
  • Tkinter UI with 1D or 2D simulation
  • Boundary modes (None, Fade, CAP, Both)
  • Environment modes (None, Thermal, Stochastic inflow, Both)
  • Controls for Morse potential parameters
  • Data export, figure saving, sensitivity analysis
  • A "base bond" arrow behind the proton

Dependencies:
  - constants.py
  - quantum_ops.py
  - animator.py
  - dna_bases.py    (for optional base data)
  - analysis.py     (for export_data, save_figure, save_all_figures)
"""

import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import constants
import quantum_ops
import dna_bases
import analysis
from animator import Animator1D, Animator2D

# -----------------------------------------------------------------------------
# Main Window
# -----------------------------------------------------------------------------
root = tk.Tk()
root.title("Quantum DNA Simulation: Proton Tunneling & Coupled Oscillators")
root.geometry("1200x800")

style = ttk.Style()
style.theme_use("clam")


# -----------------------------------------------------------------------------
# Top Frame: Display Mode (1D or 2D) + Menu
# -----------------------------------------------------------------------------
frame_top = ttk.Frame(root, padding=10)
frame_top.grid(row=0, column=0, sticky="ew")

ttk.Label(frame_top, text="Select Display Mode:", font=("Arial", 14)).grid(
    row=0, column=0, sticky="w", padx=5
)

var_display = tk.StringVar(value="1D")
ttk.Radiobutton(
    frame_top, text="1D Simulation", variable=var_display, value="1D"
).grid(row=0, column=1, padx=10)
ttk.Radiobutton(
    frame_top, text="2D Simulation", variable=var_display, value="2D"
).grid(row=0, column=2, padx=10)

menu_bar = tk.Menu(root)
file_menu = tk.Menu(menu_bar, tearoff=0)

def export_data_callback():
    # Exports 1D data (adapt if you want 2D as well).
    if 'animator1D' in globals():
        analysis.export_data(animator1D.data['time'], animator1D.data['energy'])

def save_figure_callback():
    # Example placeholder for saving a single figure
    pass

def save_all_figures_callback():
    # Saves all open matplotlib figures
    analysis.save_all_figures()

#
# Minimal working sensitivity example:
#
def run_sensitivity_callback():
    """
    A minimal example sensitivity analysis that varies 'D' and prints/saves results.
    """
    # We'll do a trivial "simulation_func" that just returns something based on D
    def dummy_simulation_func(D=0.2, a=1.5e10, x0=0.5e-10, C=0.15):
        # Realistically you'd run a short sim, measure final energy, etc.
        return D**2  # trivial example

    param_name = "D"
    param_values = [0.1, 0.2, 0.3, 0.4]
    results = analysis.sensitivity_analysis(
        simulation_func=dummy_simulation_func,
        param_name=param_name,
        param_values=param_values
    )
    # Save to CSV
    analysis.export_sensitivity_results(results, "sensitivity_results.csv")
    print("Sensitivity analysis completed. Results saved to sensitivity_results.csv")

file_menu.add_command(label="Export Data", command=export_data_callback)
file_menu.add_command(label="Save Figure", command=save_figure_callback)
file_menu.add_command(label="Save All Figures", command=save_all_figures_callback)
file_menu.add_command(label="Sensitivity Analysis", command=run_sensitivity_callback)

menu_bar.add_cascade(label="File", menu=file_menu)
root.config(menu=menu_bar)


# -----------------------------------------------------------------------------
# Main Content Frame (for 1D and 2D)
# -----------------------------------------------------------------------------
frame_main = ttk.Frame(root, padding=10)
frame_main.grid(row=1, column=0, sticky="nsew")
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

frame_1D = ttk.Frame(frame_main, padding=5)
frame_2D = ttk.Frame(frame_main, padding=5)
frame_1D.grid(row=0, column=0, sticky="nsew")
frame_2D.grid_forget()  # hidden initially


# -----------------------------------------------------------------------------
# 1D Simulation Setup
# -----------------------------------------------------------------------------
x = np.linspace(constants.X_MIN_1D, constants.X_MAX_1D, constants.N_1D)
dx = x[1] - x[0]
kx = 2 * np.pi * np.fft.fftfreq(constants.N_1D, d=dx)

# Initial wavefunction
x_center = -0.8e-10
sigma = 0.1e-10
psi_init_1D = np.exp(-((x - x_center)**2) / (2*sigma**2))
psi_init_1D /= np.sqrt(np.sum(np.abs(psi_init_1D)**2)*dx)

# Data storage arrays
time_list = []
energy_list = []
pleft_list = []
pright_list = []

def compute_energy_1D(psi, V):
    psi_k = np.fft.fft(psi)
    kin_k = (constants.HBAR**2)*(kx**2)/(2*constants.M_PROTON)
    kin = np.sum(np.abs(psi_k)**2 * kin_k) / constants.N_1D
    pot = np.sum(np.abs(psi)**2 * V) * dx
    return kin + pot

def prob_left_right(psi):
    mid_ind = np.searchsorted(x, 0.0)
    p_left = np.sum(np.abs(psi[:mid_ind])**2)*dx
    p_right = np.sum(np.abs(psi[mid_ind:])**2)*dx
    return p_left, p_right

# 1D UI variables
ui_vars_1D = {
    'D': tk.DoubleVar(value=0.2),
    'a': tk.DoubleVar(value=1.5e10),
    'x0': tk.DoubleVar(value=0.5e-10),
    'C': tk.DoubleVar(value=0.15),
    'T': tk.DoubleVar(value=300),
    'R': tk.DoubleVar(value=0.0),
    'Boundary': tk.StringVar(value="none"),    # "none", "fade", "cap", "both"
    'Environment': tk.StringVar(value="none"), # "none", "therm", "stoch", "both"
    'CAPalpha': tk.DoubleVar(value=1.0e8),     # a less aggressive CAP by default
    'Inflow': tk.DoubleVar(value=1e10),
    'X_MIN_1D': tk.DoubleVar(value=constants.X_MIN_1D),
    'X_MAX_1D': tk.DoubleVar(value=constants.X_MAX_1D),
    'ConstantT': tk.BooleanVar(value=False)
}

# 1D Control Panel
frame_ctrl_1D = ttk.LabelFrame(frame_1D, text="1D Simulation Controls", padding=10)
frame_ctrl_1D.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

ttk.Label(frame_ctrl_1D, text="D (eV):").grid(row=0, column=0, sticky="w")
scale_D = tk.Scale(frame_ctrl_1D, variable=ui_vars_1D['D'], from_=0.01, to=1.0, resolution=0.01,
                   orient=tk.HORIZONTAL, length=150)
scale_D.grid(row=0, column=1, padx=5, pady=2)

ttk.Label(frame_ctrl_1D, text="a (1/m):").grid(row=1, column=0, sticky="w")
scale_a = tk.Scale(frame_ctrl_1D, variable=ui_vars_1D['a'], from_=0.5e10, to=3e10, resolution=0.1e10,
                   orient=tk.HORIZONTAL, length=150)
scale_a.grid(row=1, column=1, padx=5, pady=2)

ttk.Label(frame_ctrl_1D, text="x0 (m):").grid(row=2, column=0, sticky="w")
scale_x0 = tk.Scale(frame_ctrl_1D, variable=ui_vars_1D['x0'], from_=0.1e-10, to=1e-10, resolution=0.05e-10,
                    orient=tk.HORIZONTAL, length=150)
scale_x0.grid(row=2, column=1, padx=5, pady=2)

ttk.Label(frame_ctrl_1D, text="C (eV):").grid(row=3, column=0, sticky="w")
scale_C = tk.Scale(frame_ctrl_1D, variable=ui_vars_1D['C'], from_=0.0, to=0.5, resolution=0.01,
                   orient=tk.HORIZONTAL, length=150)
scale_C.grid(row=3, column=1, padx=5, pady=2)

ttk.Label(frame_ctrl_1D, text="Temperature (K):").grid(row=0, column=2, sticky="w")
scale_T = tk.Scale(frame_ctrl_1D, variable=ui_vars_1D['T'], from_=1, to=1000, resolution=1,
                   orient=tk.HORIZONTAL, length=150)
scale_T.grid(row=0, column=3, padx=5, pady=2)

ttk.Label(frame_ctrl_1D, text="Radiation:").grid(row=1, column=2, sticky="w")
scale_R = tk.Scale(frame_ctrl_1D, variable=ui_vars_1D['R'], from_=0.0, to=1.0, resolution=0.05,
                   orient=tk.HORIZONTAL, length=150)
scale_R.grid(row=1, column=3, padx=5, pady=2)

# Boundary mode
ttk.Label(frame_ctrl_1D, text="Boundary:").grid(row=4, column=0, sticky="w")
for i, mode in enumerate(["none", "fade", "cap", "both"]):
    rb = ttk.Radiobutton(frame_ctrl_1D, text=mode.capitalize(),
                         variable=ui_vars_1D['Boundary'], value=mode)
    rb.grid(row=4, column=i+1, padx=2, pady=2)

# Environment mode
ttk.Label(frame_ctrl_1D, text="Environment:").grid(row=5, column=0, sticky="w")
for i, mode in enumerate(["none", "therm", "stoch", "both"]):
    rb = ttk.Radiobutton(frame_ctrl_1D, text=mode.capitalize(),
                         variable=ui_vars_1D['Environment'], value=mode)
    rb.grid(row=5, column=i+1, padx=2, pady=2)

ttk.Label(frame_ctrl_1D, text="CAP alpha:").grid(row=6, column=0, sticky="w")
scale_CAP = tk.Scale(frame_ctrl_1D, variable=ui_vars_1D['CAPalpha'], from_=0.0, to=1e10, resolution=1e9,
                     orient=tk.HORIZONTAL, length=150)
scale_CAP.grid(row=6, column=1, padx=5, pady=2)

ttk.Label(frame_ctrl_1D, text="Inflow:").grid(row=6, column=2, sticky="w")
scale_inflow = tk.Scale(frame_ctrl_1D, variable=ui_vars_1D['Inflow'], from_=0.0, to=1e12, resolution=1e10,
                        orient=tk.HORIZONTAL, length=150)
scale_inflow.grid(row=6, column=3, padx=5, pady=2)

chk_constT = ttk.Checkbutton(frame_ctrl_1D, text="Constant T Inflow", variable=ui_vars_1D['ConstantT'])
chk_constT.grid(row=7, column=0, columnspan=2, sticky="w", padx=5, pady=2)

# Optional: DNA base selection
ttk.Label(frame_ctrl_1D, text="Select Base (A/T/C/G):").grid(row=8, column=0, sticky="w")
entry_base = ttk.Entry(frame_ctrl_1D, width=5)
entry_base.grid(row=8, column=1, padx=5, pady=2)

def on_select_base():
    """
    Updates potential parameters AND updates the base-pair graphic
    to show the correct bases (A-T, G-C, etc.).
    """
    base = entry_base.get().strip().upper()
    if base in dna_bases.BASE_DATA:
        ui_vars_1D['D'].set(dna_bases.BASE_DATA[base]['D'])
        ui_vars_1D['a'].set(dna_bases.BASE_DATA[base]['a'])
        ui_vars_1D['x0'].set(dna_bases.BASE_DATA[base]['x0'])
        ui_vars_1D['C'].set(dna_bases.BASE_DATA[base]['C'])

        # Also update the base-pair figure to reflect the new base & complement
        comp = dna_bases.get_complement(base)
        ax_base.clear()
        ax_base.set_title("Base Pair", fontsize=12)
        ax_base.set_xlim(0, 12)
        ax_base.set_ylim(0, 6)
        ax_base.set_xticks([])
        ax_base.set_yticks([])
        # Re-draw the arrow
        ax_base.arrow(4.5, 3, 3, 0, head_width=0.1, length_includes_head=True, color='k', zorder=3)
        # Re-draw the proton circle
        ax_base.add_patch(circle_proton)
        circle_proton.center = (6, 3)
        # Draw the new base labels
        ax_base.text(3, 3, base, fontsize=14, color='blue', ha='center', va='center', zorder=5)
        ax_base.text(9, 3, comp, fontsize=14, color='blue', ha='center', va='center', zorder=5)
        canvas_base.draw_idle()
    else:
        print("Invalid base.")

btn_base = ttk.Button(frame_ctrl_1D, text="Set Base", command=on_select_base)
btn_base.grid(row=8, column=2, padx=5, pady=2)

def reset_psi_1D():
    # Reset the 1D wavefunction and clear time, energy, etc.
    animator1D.psi[:] = psi_init_1D
    time_list.clear()
    energy_list.clear()
    pleft_list.clear()
    pright_list.clear()
    animator1D.frame_counter = 0

btn_reset_psi = ttk.Button(frame_ctrl_1D, text="Reset Psi", command=reset_psi_1D)
btn_reset_psi.grid(row=8, column=3, padx=5, pady=2)


# -----------------------------------------------------------------------------
# 1D Plots
# -----------------------------------------------------------------------------
frame_plots_1D = ttk.Frame(frame_1D, padding=5)
frame_plots_1D.grid(row=1, column=0, sticky="nsew")
frame_1D.rowconfigure(1, weight=1)
frame_1D.columnconfigure(0, weight=1)

fig_pot, ax_pot = plt.subplots(figsize=(3,2))
fig_psi, ax_psi = plt.subplots(figsize=(3,2))
fig_en, ax_en = plt.subplots(figsize=(3,2))
fig_prob, ax_prob = plt.subplots(figsize=(3,2))
fig_base, ax_base = plt.subplots(figsize=(3,2))

canvas_pot = FigureCanvasTkAgg(fig_pot, master=frame_plots_1D)
canvas_psi = FigureCanvasTkAgg(fig_psi, master=frame_plots_1D)
canvas_en  = FigureCanvasTkAgg(fig_en, master=frame_plots_1D)
canvas_prob = FigureCanvasTkAgg(fig_prob, master=frame_plots_1D)
canvas_base = FigureCanvasTkAgg(fig_base, master=frame_plots_1D)

canvas_pot.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
canvas_psi.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)
canvas_base.get_tk_widget().grid(row=0, column=2, padx=5, pady=5)
canvas_en.get_tk_widget().grid(row=1, column=0, padx=5, pady=5)
canvas_prob.get_tk_widget().grid(row=1, column=1, padx=5, pady=5)

# Increase font size for ticks so they're visible
ax_pot.tick_params(labelsize=10)
ax_psi.tick_params(labelsize=10)
ax_en.tick_params(labelsize=10)
ax_prob.tick_params(labelsize=10)
ax_base.tick_params(labelsize=10)

ax_pot.set_title("Potential", fontsize=12)
ax_pot.set_xlabel("x (Å)", fontsize=10)
ax_pot.set_ylabel("V (eV)", fontsize=10)

ax_psi.set_title("Wavefunction", fontsize=12)
ax_psi.set_xlabel("x (Å)", fontsize=10)
ax_psi.set_ylabel("|Ψ|²", fontsize=10)

ax_en.set_title("Energy vs Time", fontsize=12)
ax_en.set_xlabel("Time (fs)", fontsize=10)
ax_en.set_ylabel("Energy (eV)", fontsize=10)

ax_prob.set_title("Tunneling Probability", fontsize=12)
ax_prob.set_xlabel("Time (fs)", fontsize=10)
ax_prob.set_ylabel("Probability", fontsize=10)

ax_base.set_title("Base Pair", fontsize=12)
ax_base.set_xlim(0,12)
ax_base.set_ylim(0,6)
ax_base.set_xticks([])
ax_base.set_yticks([])

ax_base.text(3,3,"A", fontsize=14, color='blue', ha='center', va='center', zorder=5)
ax_base.text(9,3,"T", fontsize=14, color='blue', ha='center', va='center', zorder=5)
ax_base.arrow(4.5,3,3,0, head_width=0.1, length_includes_head=True, color='k', zorder=3)

circle_proton = plt.Circle((6,3), radius=0.3, color='orange', zorder=5)
ax_base.add_patch(circle_proton)

line_pot, = ax_pot.plot(x*1e10, np.zeros_like(x), 'k-')

# -----------------------------------------------------------------------------
# Instantiate Animator1D (before the proton loop calls it)
# -----------------------------------------------------------------------------
data_1D = {
    'time': time_list,
    'energy': energy_list,
    'pleft': pleft_list,
    'pright': pright_list,
    'compute_energy': compute_energy_1D,
    'prob_left_right': prob_left_right
}
plots_1D = {
    'ax_pot': ax_pot,
    'line_pot': line_pot,
    'ax_psi': ax_psi,
    'canvas_psi': canvas_psi,
    'ax_en': ax_en,
    'canvas_en': canvas_en,
    'ax_prob': ax_prob,
    'canvas_prob': canvas_prob,
    #
    # <-- Added so we can force potential plot to refresh in real-time:
    #
    'canvas_pot': canvas_pot
}

animator1D = Animator1D(
    psi_init_1D, x, dx, kx,
    constants.DEFAULT_DT, constants.DEFAULT_STEPS_PER_FRAME,
    ui_vars_1D, plots_1D, data_1D,
    callback_after=root.after
)

def update_proton_position_main(psi, x_array, dx_val, circle_obj, canvas_obj):
    psi_sq = np.abs(psi)**2
    x_mean = np.sum(x_array * psi_sq) * dx_val
    frac = (x_mean - constants.X_MIN_1D) / (constants.X_MAX_1D - constants.X_MIN_1D)
    frac = max(0, min(1, frac))
    x_disp = 3 + 6*frac
    circle_obj.center = (x_disp, 3)
    canvas_obj.draw_idle()

def update_proton_loop():
    if animator1D.anim_running:
        update_proton_position_main(animator1D.psi, x, dx, circle_proton, canvas_base)
    root.after(50, update_proton_loop)

update_proton_loop()

# -----------------------------------------------------------------------------
# 2D Simulation Setup
# -----------------------------------------------------------------------------
twoD_x = np.linspace(constants.X_MIN_2D, constants.X_MAX_2D, constants.N_2D)
twoD_dx = twoD_x[1] - twoD_x[0]
twoD_X1, twoD_X2 = np.meshgrid(twoD_x, twoD_x)

kx_2D = 2 * np.pi * np.fft.fftfreq(constants.N_2D, d=twoD_dx)
ky_2D = 2 * np.pi * np.fft.fftfreq(constants.N_2D, d=twoD_dx)
KX_2D, KY_2D = np.meshgrid(kx_2D, ky_2D)

twoD_x_center = -0.5e-10
twoD_sigma = 0.2e-10
psi_init_2D = np.exp(
    -((twoD_X1 - twoD_x_center)**2 + (twoD_X2 + twoD_x_center)**2) / (2*(twoD_sigma**2))
)
norm_2D = np.sqrt(np.sum(np.abs(psi_init_2D)**2)*twoD_dx*twoD_dx)
psi_init_2D /= norm_2D

ui_vars_2D = {
    'D': tk.DoubleVar(value=0.2),
    'a': tk.DoubleVar(value=1.5e10),
    'x0': tk.DoubleVar(value=0.5e-10),
    'C': tk.DoubleVar(value=0.15),
    'T': tk.DoubleVar(value=300),
    'R': tk.DoubleVar(value=0.0),
    'Boundary': tk.StringVar(value="none"),    # "none", "fade", "cap", "both"
    'Environment': tk.StringVar(value="none"), # "none", "therm", "stoch", "both"
    'CAPalpha': tk.DoubleVar(value=1e8),
    'Inflow': tk.DoubleVar(value=1e10),
    'X_MIN_2D': tk.DoubleVar(value=constants.X_MIN_2D),
    'X_MAX_2D': tk.DoubleVar(value=constants.X_MAX_2D)
}

frame_ctrl_2D = ttk.LabelFrame(frame_2D, text="2D Simulation Controls", padding=10)
frame_ctrl_2D.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

ttk.Label(frame_ctrl_2D, text="D (eV):").grid(row=0, column=0, sticky="w")
scale2D_D = tk.Scale(
    frame_ctrl_2D, variable=ui_vars_2D['D'], from_=0.01, to=1.0, resolution=0.01,
    orient=tk.HORIZONTAL, length=150
)
scale2D_D.grid(row=0, column=1, padx=5, pady=2)

ttk.Label(frame_ctrl_2D, text="CAP alpha:").grid(row=0, column=2, sticky="w")
scale2D_CAP = tk.Scale(
    frame_ctrl_2D, variable=ui_vars_2D['CAPalpha'], from_=0.0, to=1e10, resolution=1e9,
    orient=tk.HORIZONTAL, length=150
)
scale2D_CAP.grid(row=0, column=3, padx=5, pady=2)

frame_plots_2D = ttk.Frame(frame_2D, padding=5)
frame_plots_2D.grid(row=1, column=0, sticky="nsew")
frame_2D.rowconfigure(1, weight=1)
frame_2D.columnconfigure(0, weight=1)

fig_2D_dynamic, ax_2D_dynamic = plt.subplots(figsize=(5,4))
canvas_2D_dynamic = FigureCanvasTkAgg(fig_2D_dynamic, master=frame_plots_2D)
canvas_2D_dynamic.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

ax_2D_dynamic.set_title("Dynamic Wavefunction (2D)", fontsize=12)
ax_2D_dynamic.set_xlabel("x1 (Å)", fontsize=10)
ax_2D_dynamic.set_ylabel("x2 (Å)", fontsize=10)

plots_2D = {
    'ax_dyn': ax_2D_dynamic,
    'canvas_dyn': canvas_2D_dynamic
}
data_2D = {}

animator2D = Animator2D(
    psi_init_2D, (twoD_X1, twoD_X2), twoD_dx, KX_2D, KY_2D,
    constants.DEFAULT_DT, constants.DEFAULT_STEPS_PER_FRAME,
    ui_vars_2D, plots_2D, data_2D,
    callback_after=root.after
)

# -----------------------------------------------------------------------------
# Toggle 1D / 2D
# -----------------------------------------------------------------------------
def update_display_mode(*args):
    mode = var_display.get()
    if mode == "1D":
        frame_1D.grid()
        frame_2D.grid_forget()
        animator2D.stop()
        animator1D.start()
    else:
        frame_1D.grid_forget()
        frame_2D.grid()
        animator1D.stop()
        animator2D.start()

var_display.trace_add("write", update_display_mode)

# -----------------------------------------------------------------------------
# Start with 1D animation
# -----------------------------------------------------------------------------
animator1D.start()
root.mainloop()
