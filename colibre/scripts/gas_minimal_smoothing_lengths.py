"""
Makes a histogram of the gas minimal smoothing lengths divided by minimum allowed gravitational softening and
split by redshifts. Uses the swiftsimio library.
"""

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load
from unyt import unyt_quantity

h_bounds = [1e-5, 5e2]  # dimensionless
number_of_bins = 256
hmin_bins = np.logspace(np.log10(h_bounds[0]), np.log10(h_bounds[1]), number_of_bins)
log_hmin_bin_width = np.log10(hmin_bins[1]) - np.log10(hmin_bins[0])
hmin_centers = 10.0 ** (0.5 * (np.log10(hmin_bins[1:]) + np.log10(hmin_bins[:-1])))


def get_data(filename):
    """
    Grabs the data (gas-particle minimal smoothing lengths and redshifts at which they have been reached).
    """

    data = load(filename)

    # Gas particles' minimal comoving smoothing lengths
    hmin_gas = data.gas.minimal_smoothing_lengths.to_comoving().to("kpc")
    hmin_scale_factors = data.gas.minimal_smoothing_length_scale_factors

    hmin_redshifts = 1 / hmin_scale_factors.value - 1

    # Minimal smoothing length in units of gravitational softening
    h_min_ratio = unyt_quantity(
        float(data.metadata.parameters.get("SPH:h_min_ratio")), units="dimensionless"
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

    # ρ(|r|) = W (|r|, 3.0 * eps_Plummer )
    softening_plummer_equivalent = 3.0

    # Compute the minimum allowed gravitational softening
    grav_softening_limit = softening_plummer_equivalent * np.minimum(
        eps_b_comov, eps_b_phys_max * (1.0 + hmin_redshifts)
    )

    hmin_gas_over_softening = hmin_gas * gamma / grav_softening_limit

    return hmin_gas_over_softening, hmin_redshifts, h_min_ratio.value


def make_single_image(filenames, names, output_path):
    """
    Makes a single histogram of the gas particle minimal smoothing lengths (plotted in units of minimum allowed
    gravitational softening length at the redshift at which a given gas particle's minimal smoothing length was
    reached.)
    """

    # Begin plotting
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    axes = axes.flat

    ax_dict = {"$z < 1$": axes[0], "$1 < z < 3$": axes[1], "$z > 3$": axes[2]}

    for label, ax in ax_dict.items():
        ax.loglog()
        ax.text(
            0.025, 1.0 - 0.025 * 3, label, transform=ax.transAxes, ha="left", va="top"
        )

    for color, (filename, name) in enumerate(zip(filenames, names)):
        hmin_gas_over_min_softening, hmin_redshifts, hmin_to_softening_ratio = get_data(
            filename
        )

        # Segment minimal smoothing lengths into redshift bins
        hmin_gas_over_min_softening_by_redshift = {
            "$z < 1$": hmin_gas_over_min_softening[hmin_redshifts < 1],
            "$1 < z < 3$": hmin_gas_over_min_softening[
                np.logical_and(hmin_redshifts > 1, hmin_redshifts < 3)
            ],
            "$z > 3$": hmin_gas_over_min_softening[hmin_redshifts > 3],
        }

        # Total number of stars formed
        Num_of_gas_total = len(hmin_redshifts)

        for redshift, ax in ax_dict.items():
            data = hmin_gas_over_min_softening_by_redshift[redshift]

            H, _ = np.histogram(data, bins=hmin_bins)
            y_points = H / log_hmin_bin_width / Num_of_gas_total

            ax.plot(hmin_centers, y_points, label=name, color=f"C{color}")

            # Add vertical line showing lowest possible hmin to softening ratio
            ax.axvline(x=hmin_to_softening_ratio, color=f"C{color}", ls="--", lw=0.2)

    axes[0].legend(loc="upper right", markerfirst=False)
    axes[2].set_xlabel(
        "Minimal Smoothing length over softening $h_{\\rm min}/\\varepsilon_{\\rm soft,min}$"
    )
    axes[1].set_ylabel(
        "$N_{\\rm bin}$ / d$\\log (h_{\\rm min}/\\varepsilon_{\\rm soft,min}$) / $N_{\\rm total}$"
    )

    fig.savefig(f"{output_path}/gas_minimal_smoothing_lengths.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(
        description="Gas-particle minimal smoothing-length histogram, in units of softening and split by redshifts."
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
        output_path=arguments.output_directory,
    )
