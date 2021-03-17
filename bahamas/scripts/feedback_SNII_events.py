"""
Makes a histogram of the gas particles' number of received SNII feedback events.
"""

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load

from unyt import unyt_quantity
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation

N_bounds = [-1, 9]


def get_data(filename):
    """
    Grabs the data
    """

    data = load(filename)

    gas_N_SNII = data.gas.heated_by_sniifeedback

    return gas_N_SNII


def make_single_image(filenames, names, N_bounds, number_of_simulations, output_path):
    """
    Makes a single histogram of the gas particle number of feedback events.
    """

    fig, ax = plt.subplots()

    ax.set_xlabel("Gas num. of SNII events received $N_{\\rm SNII}$ [-]")
    ax.set_ylabel("PDF [-]")
    ax.semilogy()

    num_bars = len(names) + 2.0
    width = 1.0 / num_bars
    count = 0

    for filename, name in zip(filenames, names):
        N_SNII = get_data(filename)
        N, bin_edges = np.histogram(
            N_SNII,
            range=N_bounds,
            bins=np.arange(np.min(N_bounds) - 0.5, np.max(N_bounds) + 0.5),
            density=True,
        )
        bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        ax.bar(
            bins - 0.5 * len(names) / num_bars + count * width,
            N,
            width=width,
            label=name,
            align="edge",
        )
        count += 1

    ax.legend()
    ax.set_xlim(*N_bounds)

    fig.savefig(f"{output_path}/gas_num_SNII_events.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(
        description="Basic gas particle number of SN feedback events."
    )

    snapshot_filenames = [
        f"{directory}/{snapshot}"
        for directory, snapshot in zip(
            arguments.directory_list, arguments.snapshot_list
        )
    ]

    plt.style.use(arguments.stylesheet_location)

    make_single_image(
        filenames=snapshot_filenames,
        names=arguments.name_list,
        N_bounds=N_bounds,
        number_of_simulations=arguments.number_of_inputs,
        output_path=arguments.output_directory,
    )
