"""
Makes a histogram of the gas smoothing lengths. Uses the swiftsimio library.
"""

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load
from unyt import unyt_quantity, dimensionless, Mpc

h_bounds = [1e-2, 5e2]  # comoving kpc


def get_data(filename):
    """
    Grabs the data (gas-particle smoothing lengths are in comoving kpc).
    """

    data = load(filename)

    h_gas = data.gas.smoothing_lengths.to("kpc")

    # Minimal smoothing length in units of gravitational softening
    h_min_ratio = unyt_quantity(
        float(data.metadata.parameters.get("SPH:h_min_ratio")), units=dimensionless,
    )

    # Comoving softening
    eps_b_comov = unyt_quantity(
        float(
            data.metadata.gravity_scheme.get(
                "Comoving baryon softening length (Plummer equivalent)  [internal units]"
            )
        ),
        units=Mpc,
    ).to("kpc")

    # Maximal physical softening
    eps_b_phys_max = unyt_quantity(
        float(
            data.metadata.gravity_scheme.get(
                "Maximal physical baryon softening length (Plummer equivalent) [internal units]"
            )
        ),
        units=Mpc,
    ).to("kpc")

    # Get gamma = H/h
    gamma = data.metadata.hydro_scheme.get("Kernel gamma")

    # Redshift of the snapshot
    z = data.metadata.redshift

    # Compute minimal comoving smoothing length in ckpc
    h_min = 3.0 * h_min_ratio.value * min(eps_b_comov, eps_b_phys_max * (1 + z)) / gamma

    return h_gas, h_min


def make_single_image(filenames, names, h_bounds, number_of_simulations, output_path):
    """
    Makes a single histogram of the gas particle smoothing lengths.
    """

    fig, ax = plt.subplots()
    ax.set_xlabel("Gas Particle Smoothing length $h$ [ckpc]")
    ax.set_ylabel("PDF [-]")
    ax.loglog()

    for filename, name in zip(filenames, names):
        h_gas, h_min = get_data(filename)
        h, bin_edges = np.histogram(h_gas, range=h_bounds, bins=250, density=True)
        bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        (line,) = ax.plot(bins, h, label=name)
        ax.axvline(x=h_min, color=line.get_color(), ls="--", lw=0.2)

    ax.legend()

    fig.savefig(f"{output_path}/gas_smoothing_lengths.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(
        description="Basic gas-particle smoothing-length histogram."
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
        h_bounds=h_bounds,
        number_of_simulations=arguments.number_of_inputs,
        output_path=arguments.output_directory,
    )
