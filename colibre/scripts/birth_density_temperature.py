"""
Makes a stellar-birth-density vs. metallicity 2D plot. Uses the swiftsimio library.
"""

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load

from unyt import mh
from matplotlib.colors import LogNorm

# Set the limits of the figure.
density_bounds = [0.01, 10 ** 7.0]  # in nh/cm^3
temperature_bounds = [10.0, 2.0e4]  # in K
bins = 128


def get_data(filename):
    """
    Grabs the data (stellar birth temperature in K and stellar birth density in mh / cm^3).
    """

    data = load(filename)

    birth_densities = data.stars.birth_densities.to("g/cm**3") / mh.to("g")
    birth_temperatures = data.stars.birth_temperatures.to("K")

    return birth_densities.value, birth_temperatures.value


def make_hist(filename, density_bounds, temperature_bounds, bins):
    """
    Makes the histogram for filename with bounds as lower, higher
    for the bins and "bins" the number of bins along each dimension.

    Also returns the edges for pcolormesh to use.
    """

    density_bins = np.logspace(
        np.log10(density_bounds[0]), np.log10(density_bounds[1]), bins
    )
    temperature_bins = np.logspace(
        np.log10(temperature_bounds[0]), np.log10(temperature_bounds[1]), bins
    )

    H, density_edges, temperature_edges = np.histogram2d(
        *get_data(filename), bins=[density_bins, temperature_bins]
    )

    return H.T, density_edges, temperature_edges


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
        vertical_number, horizontal_number, squeeze=True, sharex=True, sharey=True
    )

    ax = np.array([ax]) if number_of_simulations == 1 else ax

    if horizontal_number * vertical_number > number_of_simulations:
        for axis in ax.flat[number_of_simulations:]:
            axis.axis("off")

    # Set all valid on bottom row to have the horizontal axis label.
    for axis in np.atleast_2d(ax)[:][-1]:
        axis.set_xlabel("$\\rho_B$ [$n_H$ cm$^{-3}$]")

    for axis in np.atleast_2d(ax).T[:][0]:
        axis.set_ylabel("$T_B$ [K]")

    ax.flat[0].loglog()

    return fig, ax


def make_single_image(
    filenames,
    names,
    number_of_simulations,
    density_bounds,
    temperature_bounds,
    bins,
    output_path,
):
    """
    Makes a single plot of rho_birth-T_birth
    """

    fig, ax = setup_axes(number_of_simulations=number_of_simulations)

    hists = []

    for filename in filenames:
        hist, d, T = make_hist(filename, density_bounds, temperature_bounds, bins)
        hists.append(hist)

    vmax = np.max([np.max(hist) for hist in hists])

    for hist, name, axis in zip(hists, names, ax.flat):
        mappable = axis.pcolormesh(d, T, hist, norm=LogNorm(vmin=1, vmax=vmax))
        axis.text(
            0.025,
            0.975,
            name,
            ha="left",
            va="top",
            transform=axis.transAxes,
            fontsize=5,
            in_layout=False,
        )

    fig.colorbar(mappable, ax=ax.ravel().tolist(), label="Number of stellar particles")

    fig.savefig(f"{output_path}/birth_density_temperature.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(
        description="Stellar-birth-density vs. Stellar-birth-temperature plot."
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
        number_of_simulations=arguments.number_of_inputs,
        density_bounds=density_bounds,
        temperature_bounds=temperature_bounds,
        bins=bins,
        output_path=arguments.output_directory,
    )
