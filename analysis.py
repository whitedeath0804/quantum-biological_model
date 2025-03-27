# analysis.py
"""
This module provides functions for exporting simulation data,
saving figures, performing sensitivity analyses, and now
saving all figures at once.
"""

import csv
import matplotlib.pyplot as plt
from tkinter import filedialog
import os

def export_data(time_array, energy_array):
    """
    Exports simulation time and energy data to a CSV file.
    """
    filename = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")]
    )
    if not filename:
        return

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time (s)", "Energy (J)"])
        for t, E in zip(time_array, energy_array):
            writer.writerow([t, E])
    print("Data exported to", filename)


def save_figure(fig):
    """
    Saves the given matplotlib figure to a PDF file.
    """
    filename = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf")]
    )
    if not filename:
        return
    fig.savefig(filename, format="pdf")
    print("Figure saved to", filename)


def sensitivity_analysis(simulation_func, param_name, param_values, **kwargs):
    """
    Runs a sensitivity analysis by varying the parameter 'param_name'.
    Returns a list of (param_value, result) tuples.
    """
    results = []
    for value in param_values:
        result = simulation_func(**{param_name: value}, **kwargs)
        results.append((value, result))
    return results


def export_sensitivity_results(results, filename):
    """
    Exports sensitivity analysis results to a CSV file.
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Result"])
        for param, result in results:
            writer.writerow([param, result])
    print("Sensitivity results exported to", filename)


# -------------------------------------------------------------------------
# New: Save all open figures at once
# -------------------------------------------------------------------------
def save_all_figures():
    """
    Save all currently open matplotlib figures to either:
      (A) separate PDFs in a chosen directory, or
      (B) one multi-page PDF.
    Uncomment whichever approach you prefer below.
    """

    # Approach A: save each figure to a separate PDF
    folder = filedialog.askdirectory()
    if not folder:
        return

    figs = [plt.figure(num) for num in plt.get_fignums()]
    for i, fig in enumerate(figs):
        outfile = os.path.join(folder, f"figure_{i}.pdf")
        fig.savefig(outfile, format="pdf")
    print(f"Saved {len(figs)} figures to separate PDFs in:", folder)

    # ---------------------------------------------------------------------
    # OR Approach B: Save them all into one multi-page PDF
    #
    # import matplotlib.backends.backend_pdf as mpdf
    # filename = filedialog.asksaveasfilename(defaultextension=".pdf",
    #                                         filetypes=[("PDF files", "*.pdf")])
    # if not filename:
    #     return
    # pdf = mpdf.PdfPages(filename)
    # figs = [plt.figure(num) for num in plt.get_fignums()]
    # for fig in figs:
    #     pdf.savefig(fig)
    # pdf.close()
    # print(f"Saved {len(figs)} figures into multi-page PDF:", filename)
