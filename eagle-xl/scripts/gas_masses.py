"""
Makes a histogram of the gas particle masses. Uses the swiftsimio library.
"""

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load

from unyt import unyt_quantity
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation

mass_bounds = [0.5, 8.5]


def get_data(filename):
    """
    Grabs the data (masses in 10**6 Msun).
    """

    data = load(filename)

    mass_gas = data.gas.masses.to("Msun") / 10 ** 6
    mass_split = unyt_quantity(
        1e4 * float(data.metadata.parameters["SPH:particle_splitting_mass_threshold"]),
        "Msun",
    )

    return mass_gas, mass_split


def setup_axes(mass_bounds, number_of_simulations: int):
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
        axis.set_xlabel("Gas Particle Masses $M_{\\rm gas}$ [10$^6$ M$_\odot$]")
        axis.set_xlim(mass_bounds)

    for axis in np.atleast_2d(ax).T[:][0]:
        axis.set_ylabel("Counts [-]")
        axis.set_yscale("log")

    return fig, ax


def make_single_image(
    filenames, names, mass_bounds, number_of_simulations, output_path
):
    """
    Makes a single histogram of the gas particle masses.
    """

    fig, ax = setup_axes(mass_bounds, number_of_simulations=number_of_simulations)

    for filename, name, axis in zip(filenames, names, ax.flat):
        m_gas, m_split = get_data(filename)
        h, bin_edges = np.histogram(m_gas, range=mass_bounds, bins=250)
        bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        axis.plot(bins, h)
        axis.axvline(x=m_split, color="k", ls="--", lw=0.2)
        axis.text(0.975, 0.975, name, ha="right", va="top", transform=axis.transAxes)

    fig.savefig(f"{output_path}/gas_masses.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(description="Basic gas particle mass histogram.")

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
        mass_bounds=mass_bounds,
        number_of_simulations=arguments.number_of_inputs,
        output_path=arguments.output_directory,
    )
