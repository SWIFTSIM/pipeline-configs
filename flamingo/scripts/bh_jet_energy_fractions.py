import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from swiftsimio import load

import unyt
from unyt import mh, cm, Gyr
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams

# Set the limits of the figure.
mass_bounds = [1e5, 3e10]
value_bounds = [0, 1]
bins = 25
def_value = -1.0
plot_scatter = False


def get_data(filename):

    data = load(filename)

    masses = data.black_holes.subgrid_masses.to("Msun")

    try:
        jet_energies = data.black_holes.injected_jet_energies
        lum_energies = data.black_holes.agntotal_injected_energies
        values = jet_energies / (jet_energies + lum_energies)

        # Take only those BHs that have either done some heating or jet feedback, otherwise the fraction is undefined.
        masses = masses[(jet_energies + lum_energies) > 0]
        values = values[(jet_energies + lum_energies) > 0]
    except:
        values = unyt.unyt_array(
            np.zeros(masses.shape), dtype=np.float64, units=unyt.dimensionless
        )

    return masses, values


def calculate_medians(filename, mass_bounds, value_bounds, bins):

    masses, values = get_data(filename)

    masses_3rd_most_massive = np.sort(masses)[-3]

    mass_bins = np.logspace(np.log10(mass_bounds[0]), np.log10(mass_bounds[1]), bins)
    bin_width = (np.log10(mass_bounds[1]) - np.log10(mass_bounds[0])) / bins

    threshold_mass = 10 ** (
        np.log10(mass_bins[mass_bins < masses_3rd_most_massive][-1]) + bin_width * 0.5
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
    ax.set_ylabel(
        "Black Hole Injected Jet Energy Fractions $E_{\\rm jet}/(E_{\\rm jet} + E_{\\rm th})$ "
    )
    ax.set_xscale("log")

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
        if number_of_simulations == 1 and plot_scatter == True:

            kwargs = dict(
                edgecolor="none", zorder=-100, facecolor=fill_plot.get_color()
            )

            # Need to "intelligently" size the markers
            kwargs["s"] = (
                rcParams["lines.markersize"]
                * (6.0 - 5.0 * np.tanh(0.75 * np.log10(masses_rest.size) - 3.0))
                / 11.0
            )

            kwargs["alpha"] = (
                5.5 - 4.5 * np.tanh(0.75 * np.log10(masses_rest.size) - 3.0)
            ) / 10.0

            ax.scatter(masses_rest, values_rest, **kwargs)

    ax.legend()
    ax.set_xlim(*mass_bounds)
    ax.set_ylim(*value_bounds)

    fig.savefig(f"{output_path}/black_hole_jet_energy_fractions.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(
        description="AGN injected jet energy fractions - BH mass relation"
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
        mass_bounds=mass_bounds,
        value_bounds=value_bounds,
        number_of_simulations=arguments.number_of_inputs,
        output_path=arguments.output_directory,
    )
