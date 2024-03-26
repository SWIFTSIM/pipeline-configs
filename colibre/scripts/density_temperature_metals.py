"""
Makes a rho-T plot normalised by the metallicity. Uses the swiftsimio library.
"""

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load

from unyt import mh, cm, unyt_array
from matplotlib.colors import Normalize

# Constants; these could be put in the parameter file but are rarely changed.
density_bounds = [10 ** (-9.5), 1e7]  # in nh/cm^3
temperature_bounds = [10 ** 0.0, 10 ** 9.5]  # in K
metallicity_bounds = [-6, -1]  # In metal mass fraction
min_metallicity = 1e-8
bins = 256


def get_data(filename):
    """
    Grabs the data (T in Kelvin, density in mh / cm^3, and log10 metallicity).
    """

    data = load(filename)

    number_density = (data.gas.densities.to_physical() / mh).to(cm ** -3)
    temperature = data.gas.temperatures.to_physical().to("K")

    metallicity = data.gas.metal_mass_fractions
    metallicity[metallicity < min_metallicity] = min_metallicity

    return number_density.value, temperature.value, np.log10(metallicity.value)


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

    dens, temps, metals = get_data(filename)

    H, density_edges, temperature_edges = np.histogram2d(
        dens, temps, bins=[density_bins, temperature_bins], weights=metals
    )

    H_norm, _, _ = np.histogram2d(dens, temps, bins=[density_bins, temperature_bins])

    # Avoid div/0
    mask = H_norm == 0.0
    H[mask] = -25
    H_norm[mask] = 1.0

    return np.ma.array((H / H_norm).T, mask=mask.T), density_edges, temperature_edges


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
        axis.set_xlabel("Density [$n_H$ cm$^{-3}$]")

    for axis in np.atleast_2d(ax).T[:][0]:
        axis.set_ylabel("Temperature [K]")

    ax.flat[0].loglog()

    return fig, ax


def plot_eos(metadata, ax):
    """
    Plots the Equation of State (Entropy Floor) and +0.3 dex in Temperature,
    which should generally enclose particles with divergent subgrid
    properties.
    """

    parameters = metadata.parameters

    for name in ["Cool", "Jeans"]:
        try:
            norm_H = float(
                parameters[f"COLIBREEntropyFloor:{name}_density_norm_H_p_cm3"]
            )
            gamma_eff = float(parameters[f"COLIBREEntropyFloor:{name}_gamma_effective"])
            norm_T = float(parameters[f"COLIBREEntropyFloor:{name}_temperature_norm_K"])
        except:
            continue

        first_point_H = 1e-10 * norm_H
        second_point_H = 1e10 * norm_H
        temp_first_point = norm_T * (first_point_H / norm_H) ** (gamma_eff - 1)
        temp_second_point = norm_T * (second_point_H / norm_H) ** (gamma_eff - 1)

        ax.plot(
            unyt_array([first_point_H, second_point_H], "cm**-3"),
            unyt_array([temp_first_point, temp_second_point], "K"),
            linestyle="dashed",
            alpha=0.5,
            color="k",
            lw=0.5,
        )

        ax.plot(
            unyt_array([first_point_H, second_point_H], "cm**-3"),
            unyt_array([temp_first_point, temp_second_point], "K") * pow(10, 0.3),
            linestyle="dotted",
            alpha=0.5,
            color="k",
            lw=0.5,
        )

    return


def make_single_image(
    filenames,
    names,
    number_of_simulations,
    density_bounds,
    temperature_bounds,
    metallicity_bounds,
    bins,
    output_path,
):
    """
    Makes a single plot of rho-T
    """

    fig, ax = setup_axes(number_of_simulations=number_of_simulations)

    hists = []

    for filename in filenames:
        hist, d, T = make_hist(filename, density_bounds, temperature_bounds, bins)
        hists.append(hist)

    for filename, hist, name, axis in zip(filenames, hists, names, ax.flat):
        mappable = axis.pcolormesh(
            d,
            T,
            hist,
            norm=Normalize(vmin=metallicity_bounds[0], vmax=metallicity_bounds[1]),
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
        metadata = load(filename).metadata
        plot_eos(metadata, axis)
        axis.set_xlim(*density_bounds)
        axis.set_ylim(*temperature_bounds)

    fig.colorbar(
        mappable,
        ax=ax.ravel().tolist(),
        label="Mean (Logarithmic) Metallicity $\log_{10} Z$ (min. $Z=10^{-8}$)",
    )

    fig.savefig(f"{output_path}/density_temperature_metals.png")

    return


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

    make_single_image(
        filenames=snapshot_filenames,
        names=arguments.name_list,
        number_of_simulations=arguments.number_of_inputs,
        density_bounds=density_bounds,
        temperature_bounds=temperature_bounds,
        metallicity_bounds=metallicity_bounds,
        bins=bins,
        output_path=arguments.output_directory,
    )
