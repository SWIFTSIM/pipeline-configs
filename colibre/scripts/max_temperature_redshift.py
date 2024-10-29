"""
Makes a maximum temperature vs. redshift 2D plot. Uses the swiftsimio library.
"""

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load
from matplotlib.colors import LogNorm

# Set the limits of the figure.
max_temperature_bounds = [1e5, 1e12]  # in K
redshift_bounds = [-1.5, 16]  # dimensionless
bins = 128


def get_data(filename):
    """
    Grabs the data (z and stellar birth density in mh / cm^3).
    """

    data = load(filename)

    # Fetch the maximal temperatures of gas and stars
    max_T_gas = data.gas.maximal_temperatures.to("K").value
    max_T_stars = data.stars.maximal_temperatures.to("K").value
    max_T = np.concatenate([max_T_gas, max_T_stars])

    # Fetch the redshifts at which the maximal temperatures were reached
    max_T_redshifts_gas = 1 / data.gas.maximal_temperature_scale_factors.value - 1
    max_T_redshifts_stars = 1 / data.stars.maximal_temperature_scale_factors.value - 1
    max_T_redshifts = np.concatenate([max_T_redshifts_gas, max_T_redshifts_stars])

    return max_T, max_T_redshifts


def make_hist(filename, max_temperature_bounds, redshift_bounds, bins):
    """
    Makes the histogram for filename with bounds as lower, higher
    for the bins and "bins" the number of bins along each dimension.

    Also returns the edges for pcolormesh to use.
    """

    T_max_bins = np.logspace(
        np.log10(max_temperature_bounds[0]), np.log10(max_temperature_bounds[1]), bins
    )
    redshift_bins = np.linspace(redshift_bounds[0], redshift_bounds[1], bins)

    H, max_temperature_edges, redshift_edges = np.histogram2d(
        *get_data(filename), bins=[T_max_bins, redshift_bins]
    )

    return H.T, max_temperature_edges, redshift_edges


def setup_axes(number_of_simulations: int):
    """
    Creates the figure and axis object. Creates a grid of a x b subplots
    that add up to at least number_of_simulations.
    """

    sqrt_number_of_simulations = np.sqrt(number_of_simulations)
    horizontal_number = int(np.ceil(sqrt_number_of_simulations))
    # Ensure >= number_of_simulations plots in a grid
    vertical_number = int(np.ceil(number_of_simulations / horizontal_number))

    fig_w, fig_h = plt.figaspect(vertical_number / horizontal_number)
    fig, ax = plt.subplots(
        vertical_number,
        horizontal_number,
        squeeze=True,
        sharex=True,
        sharey=True,
        figsize=(fig_w, fig_h),
    )

    ax = np.array([ax]) if number_of_simulations == 1 else ax

    if horizontal_number * vertical_number > number_of_simulations:
        for axis in ax.flat[number_of_simulations:]:
            axis.axis("off")

    # Set all valid on bottom row to have the horizontal axis label.
    for axis in np.atleast_2d(ax)[:][-1]:
        axis.set_xlabel("$T_{\\rm max}$ [K]")

    for axis in np.atleast_2d(ax).T[:][0]:
        axis.set_ylabel("Redshift $z$ (at $T_{\\rm max}$)")

    ax.flat[0].set_xscale("log")
    ax.flat[0].invert_yaxis()

    return fig, ax


def make_single_image(
    filenames,
    names,
    number_of_simulations,
    max_temperature_bounds,
    redshift_bounds,
    bins,
    output_path,
):
    """
    Makes a single plot of T_max-z
    """

    fig, ax = setup_axes(number_of_simulations=number_of_simulations)

    hists = []

    for filename in filenames:
        hist, max_temp, z = make_hist(
            filename, max_temperature_bounds, redshift_bounds, bins
        )
        hists.append(hist)

    vmax = np.max([np.max(hist) for hist in hists])

    for hist, name, axis in zip(hists, names, ax.flat):
        mappable = axis.pcolormesh(max_temp, z, hist, norm=LogNorm(vmin=1, vmax=vmax))
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

    fig.colorbar(mappable, ax=ax.ravel().tolist(), label="Number of particles")

    fig.savefig(f"{output_path}/max_temperature_redshift.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(
        description="Max temperature vs. redshift phase plot."
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
        max_temperature_bounds=max_temperature_bounds,
        redshift_bounds=redshift_bounds,
        bins=bins,
        output_path=arguments.output_directory,
    )
