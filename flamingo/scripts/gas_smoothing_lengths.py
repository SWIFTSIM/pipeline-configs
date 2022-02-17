"""
Makes a histogram of the gas smoothing lengths. Uses the swiftsimio library.
"""

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load
from unyt import unyt_quantity, dimensionless, Mpc

h_bounds = [1e-2, 9e2]  # comoving kpc


def get_data(filename):
    """
    Grabs the data (gas-particle smoothing lengths in comoving kpc).
    """

    data = load(filename)

    # Gas particles' comoving smoothing lengths
    h_gas = data.gas.smoothing_lengths.to_comoving().to("kpc")

    # Minimal smoothing length in units of gravitational softening
    h_min_ratio = unyt_quantity(
        float(data.metadata.parameters.get("SPH:h_min_ratio")), units=dimensionless
    )

    # Comoving softening length (Plummer equivalent)
    eps_b_comov = unyt_quantity(
        float(
            data.metadata.gravity_scheme.get(
                "Comoving baryon softening length (Plummer equivalent)  [internal units]",
                0.0,
            )
        ),
        units=data.metadata.units.length,
    ).to("kpc")

    # Maximal physical softening length (Plummer equivalent)
    eps_b_phys_max = unyt_quantity(
        float(
            data.metadata.gravity_scheme.get(
                "Maximal physical baryon softening length (Plummer equivalent) [internal units]",
                0.0,
            )
        ),
        units=data.metadata.units.length,
    ).to("kpc")

    # Get gamma = Kernel size / Kernel smoothing length
    gamma = data.metadata.hydro_scheme.get("Kernel gamma")

    # Redshift of the snapshot
    z = data.metadata.redshift

    # ρ(|r|) = W (|r|, 3.0 * eps_Plummer )
    softening_plummer_equivalent = 3.0

    # Compute the minimal comoving smoothing length in ckpc
    h_min = (
        softening_plummer_equivalent
        * h_min_ratio.value
        * min(eps_b_comov, eps_b_phys_max * (1.0 + z))
        / gamma
    )

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
        h, bin_edges = np.histogram(
            np.log10(h_gas), range=np.log10(h_bounds), bins=250, density=True
        )

        bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bins = 10**bins

        (line,) = ax.plot(bins, h, label=name)

        # Add h_min vertical line
        ax.axvline(x=h_min, color=line.get_color(), ls="--", lw=0.2)

    ax.legend()
    ax.set_xlim(*h_bounds)

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
