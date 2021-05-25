"""
Creates a plot of birth density (horizontal) against f_E. The
scatter in f_E comes from the dependence on metallicity.
"""

import matplotlib.pyplot as plt
import numpy as np
import unyt

from swiftsimio import load, SWIFTDataset

from unyt import mh, cm, Gyr
from matplotlib.colors import LogNorm, Normalize
from matplotlib.animation import FuncAnimation

number_of_bins = 128


def setup_axes(number_of_simulations: int):
    """
    Creates the figure and axis object. Creates a grid of a x b subplots
    that add up to at least number_of_simulations.
    """

    sqrt_number_of_simulations = np.sqrt(number_of_simulations)
    horizontal_number = int(np.ceil(sqrt_number_of_simulations))
    # Ensure >= number_of_simulations plots in a grid
    vertical_number = int(np.ceil(number_of_simulations / horizontal_number))

    fig, ax = plt.subplots(
        vertical_number, horizontal_number, squeeze=True, sharex=True, sharey=True,
    )

    ax = np.array([ax]) if number_of_simulations == 1 else ax

    if horizontal_number * vertical_number > number_of_simulations:
        for axis in ax.flat[number_of_simulations:]:
            axis.axis("off")

    # Set all valid on bottom row to have the horizontal axis label.
    for axis in np.atleast_2d(ax)[:][-1]:
        axis.set_xlabel("Birth Density [$n_H$ cm$^{-3}$]")

    for axis in np.atleast_2d(ax).T[:][0]:
        axis.set_ylabel("Feedback energy fraction $f_E$ []")

    ax.flat[0].loglog()

    return fig, ax


def bin_individual_data(data: SWIFTDataset):
    f_E_fractions = data.stars.feedback_energy_fractions.value
    mask = f_E_fractions > 0.0

    f_E_fractions = f_E_fractions[mask]
    birth_densities = (data.stars.birth_densities[mask] / mh).to(1 / cm ** 3).value

    birth_density_bins = unyt.unyt_array(
        np.logspace(-3, 5, number_of_bins), units=1 / cm ** 3
    )
    feedback_energy_fraction_bins = unyt.unyt_array(
        np.logspace(-2, 1, number_of_bins), units="dimensionless"
    )

    H, density_edges, f_E_edges = np.histogram2d(
        birth_densities,
        f_E_fractions,
        bins=[birth_density_bins, feedback_energy_fraction_bins],
    )

    return H, density_edges, f_E_edges


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(description="Basic density-temperature figure.")

    snapshot_filenames = [
        f"{directory}/{snapshot}"
        for directory, snapshot in zip(
            arguments.directory_list, arguments.snapshot_list
        )
    ]

    plt.style.use(arguments.stylesheet_location)

    data = [load(filename) for filename in snapshot_filenames]

    # Begin plotting
    fig, axes = setup_axes(number_of_simulations=arguments.number_of_inputs)

    # Do this first so we can ensure that we have correct vmin, vmax:
    histograms = []
    for snapshot in data:
        H, density_edges, f_E_edges = bin_individual_data(snapshot)
        histograms.append(H)

    vmax = max([H.max() for H in histograms])

    for ax, name, snapshot, H in zip(axes.flat, arguments.name_list, data, histograms):
        mappable = ax.pcolormesh(
            density_edges, f_E_edges, H.T, norm=LogNorm(vmin=1, vmax=vmax, clip=True)
        )
        f_E_fractions = snapshot.stars.feedback_energy_fractions.value
        f_E_fractions = f_E_fractions[f_E_fractions > 0.0]

        ax.text(
            0.025,
            0.025,
            "\n".join(
                [
                    "$f_E$ values:",
                    f"Min: {np.min(f_E_fractions):3.3f}",
                    f"Max: {np.max(f_E_fractions):3.3f}",
                    f"Mean: {np.mean(f_E_fractions):3.3f}",
                    f"Median: {np.median(f_E_fractions):3.3f}",
                ]
            ),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
        )

    fig.colorbar(mappable, label="Number of particles")

    fig.savefig(f"{arguments.output_directory}/birth_density_f_E.png")

