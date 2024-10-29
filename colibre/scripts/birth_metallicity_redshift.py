"""
Makes a stellar birth metallicity vs. redshift 2D plot. Uses the swiftsimio library.
"""

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load

from matplotlib.colors import LogNorm

# Set the limits of the figure.
metal_mass_fraction_bounds = [1e-4, 0.5]  # dimensionless
redshift_bounds = [-1.5, 16]  # dimensionless
bins = 128


def get_data(filename):
    """
    Grabs the data (redshifts and MMF).
    """

    data = load(filename)

    birth_redshifts = 1 / data.stars.birth_scale_factors.value - 1

    try:
        metal_mass_fractions = data.stars.smoothed_metal_mass_fractions.value
    except AttributeError:
        metal_mass_fractions = data.stars.metal_mass_fractions.value

    below_Z_min = np.where(metal_mass_fractions < metal_mass_fraction_bounds[0])

    # Stars with Z < lowest Z in the figure should be added to the lowest-Z bin
    metal_mass_fractions[below_Z_min] = metal_mass_fraction_bounds[0] * (
        1.0 + 1e-3 / bins
    )

    return metal_mass_fractions, birth_redshifts


def make_hist(filename, metal_mass_fraction_bounds, redshift_bounds, bins):
    """
    Makes the histogram for filename with bounds as lower, higher
    for the bins and "bins" the number of bins along each dimension.

    Also returns the edges for pcolormesh to use.
    """

    metal_mass_fraction_bins = np.logspace(
        np.log10(metal_mass_fraction_bounds[0]),
        np.log10(metal_mass_fraction_bounds[1]),
        bins,
    )
    redshift_bins = np.linspace(redshift_bounds[0], redshift_bounds[1], bins)

    H, metal_mass_fraction_edges, redshift_edges = np.histogram2d(
        *get_data(filename), bins=[metal_mass_fraction_bins, redshift_bins]
    )

    return H.T, metal_mass_fraction_edges, redshift_edges


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
        axis.set_xlabel("Smoothed MMF $Z$")

    for axis in np.atleast_2d(ax).T[:][0]:
        axis.set_ylabel("Redshift $z$")

    ax.flat[0].set_xscale("log")
    ax.flat[0].invert_yaxis()

    return fig, ax


def make_single_image(
    filenames,
    names,
    number_of_simulations,
    metal_mass_fraction_bounds,
    redshift_bounds,
    bins,
    output_path,
):
    """
    Makes a single plot of MMF-z
    """

    fig, ax = setup_axes(number_of_simulations=number_of_simulations)

    hists = []

    for filename in filenames:
        hist, Z, z = make_hist(
            filename, metal_mass_fraction_bounds, redshift_bounds, bins
        )
        hists.append(hist)

    vmax = np.max([np.max(hist) for hist in hists])

    for hist, name, axis in zip(hists, names, ax.flat):
        mappable = axis.pcolormesh(Z, z, hist, norm=LogNorm(vmin=1, vmax=vmax))
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

    fig.savefig(f"{output_path}/metallicity_redshift.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(
        description="Stellar-metallicity vs. redshift phase plot."
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
        metal_mass_fraction_bounds=metal_mass_fraction_bounds,
        redshift_bounds=redshift_bounds,
        bins=bins,
        output_path=arguments.output_directory,
    )
