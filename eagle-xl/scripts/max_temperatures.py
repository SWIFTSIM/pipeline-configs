"""
Makes a histogram of the gas particle max temperature. Uses the swiftsimio library.
"""

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load

from unyt import unyt_quantity
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation

T_bounds = [1e5, 3e10]


def get_data(filename):
    """
    Grabs the data
    """

    data = load(filename)

    gas_max_T = data.gas.maximal_temperatures.to("K")

    return gas_max_T


def setup_axes(T_bounds, number_of_simulations: int):
    """
    Creates the figure and axis object. Creates a grid of a x b subplots
    that add up to at least number_of_simulations.
    """

    sqrt_number_of_simulations = np.sqrt(number_of_simulations)
    horizontal_number = int(np.ceil(sqrt_number_of_simulations))
    # Ensure >= number_of_simulations plots in a grid
    vertical_number = int(np.ceil(number_of_simulations / horizontal_number))

    fig, ax = plt.subplots(
        vertical_number, horizontal_number, squeeze=True, sharex=True, sharey=True
    )

    ax = np.array([ax]) if number_of_simulations == 1 else ax

    if horizontal_number * vertical_number > number_of_simulations:
        for axis in ax.flat[number_of_simulations:]:
            axis.axis("off")

    # Set all valid on bottom row to have the horizontal axis label.
    for axis in np.atleast_2d(ax)[:][-1]:
        axis.set_xlabel("Gas Max. Temperature $T_{\\rm max}$ [K]")
        axis.set_xscale("log")
        axis.set_xlim(T_bounds)

    for axis in np.atleast_2d(ax).T[:][0]:
        axis.set_ylabel("Counts [-]")
        axis.set_yscale("log")

    return fig, ax


def make_single_image(filenames, names, T_bounds, number_of_simulations, output_path):
    """
    Makes a single histogram of the gas particle max temperatures.
    """

    fig, ax = setup_axes(T_bounds, number_of_simulations=number_of_simulations)

    for filename, name, axis in zip(filenames, names, ax.flat):
        T_max = get_data(filename)
        h, bin_edges = np.histogram(np.log10(T_max), range=np.log10(T_bounds), bins=250)
        bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bins = 10 ** bins
        axis.plot(bins, h)
        axis.text(0.975, 0.975, name, ha="right", va="top", transform=axis.transAxes)

    fig.savefig(f"{output_path}/gas_max_temperatures.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(
        description="Basic gas particle max temperature histogram."
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
        T_bounds=T_bounds,
        number_of_simulations=arguments.number_of_inputs,
        output_path=arguments.output_directory,
    )
