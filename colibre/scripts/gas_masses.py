"""
Makes a histogram of the gas particle masses. Uses the swiftsimio library.
"""

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load

from unyt import unyt_quantity
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation


def get_data(filename):
    """
    Grabs the data (masses in 10**5 Msun).
    """

    data = load(filename)

    mass_gas = data.gas.masses.to("1e5 * Msun").value
    mass_split = (
        unyt_quantity(
            float(
                data.metadata.parameters.get(
                    "SPH:particle_splitting_mass_threshold", 0.0
                )
            ),
            units=data.units.mass,
        )
        .to("1e5 * Msun")
        .value
    )

    return mass_gas, mass_split


def make_single_image(filenames, names, number_of_simulations, output_path):
    """
    Makes a single histogram of the gas particle masses.
    """

    fig, ax = plt.subplots()
    ax.set_xlabel("Gas Particle Masses $M_{\\rm gas}$ [10$^5$ M$_\\odot$]")
    ax.set_ylabel("PDF [-]")
    ax.semilogy()

    # Calculate mass bounds
    min_mass = float("inf")
    max_mass = 0
    data = {}
    for filename, name in zip(filenames, names):
        mass_gas, mass_split = get_data(filename)
        min_mass = min(min_mass, np.min(mass_gas))
        max_mass = max(max_mass, mass_split)
        data[name] = (mass_gas, mass_split)
    mass_bounds = [min_mass, max_mass]

    for name in names:
        m_gas, m_split = data[name]
        h, bin_edges = np.histogram(m_gas, range=mass_bounds, bins=250, density=True)
        bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        (line,) = ax.plot(bins, h, label=name)
        ax.axvline(x=m_split, color=line.get_color(), ls="--", lw=0.2)

    ax.legend()

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
        number_of_simulations=arguments.number_of_inputs,
        output_path=arguments.output_directory,
    )
