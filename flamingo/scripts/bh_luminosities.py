import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from swiftsimio import load

import unyt
from unyt import mh, cm, Gyr, speed_of_light
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation

# Set the limits of the figure.
mass_bounds = [1e5, 3e10]
value_bounds = [1e38, 1e48]
bins = 25
def_value = -1.0


def get_data(filename):

    data = load(filename)

    masses = data.black_holes.subgrid_masses.to("Msun")
    rad_effs = data.black_holes.radiative_efficiencies
    accr_rates = data.black_holes.accretion_rates.convert_to_units(unyt.kg / unyt.s)
    values = rad_effs * speed_of_light ** 2 * accr_rates
    values = values.convert_to_units(unyt.erg / unyt.s)

    # Make sure to take only black holes with a non-zero luminosity
    return masses[rad_effs > 1e-6], values[rad_effs > 1e-6]


def calculate_medians(filename, mass_bounds, value_bounds, bins):

    masses, values = get_data(filename)

    masses_10th_most_massive = np.sort(masses)[-7]

    mass_bins = np.logspace(np.log10(mass_bounds[0]), np.log10(mass_bounds[1]), bins)
    bin_width = (np.log10(mass_bounds[1]) - np.log10(mass_bounds[0])) / bins

    threshold_mass = 10 ** (
        np.log10(mass_bins[mass_bins < masses_10th_most_massive][-1]) + bin_width * 0.5
    )
    mass_bins = mass_bins[mass_bins < threshold_mass]

    values_most_massive = values[masses > threshold_mass]
    masses_most_massive = masses[masses > threshold_mass]

    values_rest = values[masses < threshold_mass]
    masses_rest = masses[masses < threshold_mass]

    medians, _, _ = stats.binned_statistic(
        masses_rest, values_rest, statistic="median", bins=mass_bins
    )
    percentile_10s, _, _ = stats.binned_statistic(
        masses_rest,
        values_rest,
        statistic=lambda x: np.percentile(x, 10.0),
        bins=mass_bins,
    )
    percentile_90s, _, _ = stats.binned_statistic(
        masses_rest,
        values_rest,
        statistic=lambda x: np.percentile(x, 90.0),
        bins=mass_bins,
    )
    mass_bins = np.array(
        [(mass_bins[i] + mass_bins[i + 1]) / 2 for i in range(np.size(medians))]
    )

    return (
        masses_most_massive,
        values_most_massive,
        masses_rest,
        values_rest,
        mass_bins,
        medians,
        percentile_10s,
        percentile_90s,
    )


def make_single_image(
    filenames, names, mass_bounds, value_bounds, number_of_simulations, output_path
):

    fig, ax = plt.subplots()

    ax.set_xlabel("Black Hole Subgrid Masses $M_{\\rm sub}$ [M$_\odot$]")
    ax.set_ylabel("Black Hole Luminosities $L_{\\rm bol}$ $[{\\rm erg}{\\rm s}^{-1}]$")
    ax.set_xscale("log")
    ax.set_yscale("log")

    for filename, name in zip(filenames, names):
        masses_most_massive, values_most_massive, masses_rest, values_rest, mass_bins, medians, percentile_10s, percentile_90s = calculate_medians(
            filename, mass_bounds, value_bounds, bins
        )
        fill_plot, = ax.plot(mass_bins, medians, label=name)
        ax.fill_between(
            mass_bins,
            percentile_10s,
            percentile_90s,
            alpha=0.2,
            facecolor=fill_plot.get_color(),
        )
        scatter_plot = ax.scatter(
            masses_most_massive, values_most_massive, facecolor=fill_plot.get_color()
        )
        if number_of_simulations == 1:
            ax.scatter(
                masses_rest,
                values_rest,
                s=0.75,
                edgecolors="none",
                marker="o",
                alpha=0.5,
                facecolor=fill_plot.get_color(),
            )

    ax.legend()
    ax.set_xlim(*mass_bounds)
    ax.set_ylim(*value_bounds)

    fig.savefig(f"{output_path}/black_hole_luminosities.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(description="AGN Luminosity - BH mass relation")

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
        mass_bounds=mass_bounds,
        value_bounds=value_bounds,
        number_of_simulations=arguments.number_of_inputs,
        output_path=arguments.output_directory,
    )