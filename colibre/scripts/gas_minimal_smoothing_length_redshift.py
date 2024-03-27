"""
Plots the gas minimal smoothing length versus redshift at which the minimal smoothing length was reached.
Uses the swiftsimio library.
"""

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load
from unyt import unyt_quantity
from matplotlib.colors import LogNorm

# Set the limits of the figure.
hmin_bounds = [1e-5, 1e3]  # in ckpc
redshift_bounds = [-1.5, 12]  # dimensionless
bins = 128


def get_data(filename):
    """
    Grabs the data (gas-particle minimal smoothing lengths and redshifts at which they have been reached).
    """

    data = load(filename)

    # Gas particles' minimal comoving smoothing lengths
    hmin_gas = data.gas.minimal_smoothing_lengths.to_comoving().to("kpc").value

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

    hmin_comoving = (
        softening_plummer_equivalent * h_min_ratio.value * eps_b_comov / gamma
    )
    hmin_phys_max = (
        softening_plummer_equivalent * h_min_ratio.value * eps_b_phys_max / gamma
    )

    return (
        hmin_gas,
        hmin_redshifts,
        hmin_comoving,
        hmin_phys_max,
        eps_b_comov,
        eps_b_phys_max,
    )


def make_hist(filename, hmin_bounds, redshift_bounds, bins):
    """
    Makes the histogram for filename with bounds as lower, higher
    for the bins and "bins" the number of bins along each dimension.

    Also returns the edges for pcolormesh to use.
    """

    hmin_bins = np.logspace(np.log10(hmin_bounds[0]), np.log10(hmin_bounds[1]), bins)
    redshift_bins = np.linspace(redshift_bounds[0], redshift_bounds[1], bins)

    hmin_gas, hmin_redshifts, hmin_comoving, hmin_phys_max, soft_comoving, soft_phys_max = get_data(
        filename
    )

    H, hmin_edges, redshift_edges = np.histogram2d(
        hmin_gas, hmin_redshifts, bins=[hmin_bins, redshift_bins]
    )

    return (
        H.T,
        hmin_edges,
        redshift_edges,
        hmin_comoving,
        hmin_phys_max,
        soft_comoving,
        soft_phys_max,
    )


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
        axis.set_xlabel("$h_{\\rm min}$ [ckpc]")

    for axis in np.atleast_2d(ax).T[:][0]:
        axis.set_ylabel("Redshift $z$ ($h_{\\rm min}$)")

    ax.flat[0].set_xscale("log")
    ax.flat[0].invert_yaxis()

    return fig, ax


def make_single_image(
    filenames,
    names,
    number_of_simulations,
    hmin_bounds,
    redshift_bounds,
    bins,
    output_path,
):
    """
    Makes a single plot of gas minimal smoothing length vs. redshift
    """

    fig, ax = setup_axes(number_of_simulations=number_of_simulations)

    hists = []

    for filename in filenames:
        hist, hmin_gas, z, hmin_comoving, hmin_phys_max, soft_comoving, soft_phys_max, = make_hist(
            filename, hmin_bounds, redshift_bounds, bins
        )
        hists.append(hist)

    vmax = np.max([np.max(hist) for hist in hists])

    for hist, name, axis in zip(hists, names, ax.flat):
        mappable = axis.pcolormesh(hmin_gas, z, hist, norm=LogNorm(vmin=1, vmax=vmax))

        redshifts = np.linspace(0, redshift_bounds[1], 64)

        allowed_hmin_evolution = np.minimum(
            hmin_comoving, hmin_phys_max * (1.0 + redshifts)
        )
        softening_evolution = np.minimum(
            soft_comoving, soft_phys_max * (1.0 + redshifts)
        )

        axis.plot(
            allowed_hmin_evolution, redshifts, zorder=10000, color="k", dashes=(4, 2)
        )
        axis.plot(
            softening_evolution, redshifts, zorder=10000, color="k", dashes=(1.5, 1.5)
        )

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

    fig.colorbar(mappable, ax=ax.ravel().tolist(), label="Number of gas particles")

    fig.savefig(f"{output_path}/gas_hmin_redshift.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(
        description="Gas minimal smoothing length vs. redshift phase plot."
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
        hmin_bounds=hmin_bounds,
        redshift_bounds=redshift_bounds,
        bins=bins,
        output_path=arguments.output_directory,
    )
