import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load
from velociraptor.autoplotter.objects import VelociraptorLine
from velociraptor.tools.lines import binned_median_line
from velociraptor.autoplotter.plot import scatter_x_against_y

import unyt

# Set the limits of the figure.
mass_bounds = [1e3, 3e10]  # Msun
value_bounds = [0, 1.02]  # dimensionless
bins = 25
plot_scatter = True


def get_data(filename):

    data = load(filename)

    parameters = data.metadata.parameters
    try:
        seed_mass = float(parameters["COLIBREAGN:subgrid_seed_mass_Msun"]) * unyt.Msun
    except AttributeError:
        seed_mass = float(parameters["SPINJETAGN:subgrid_seed_mass_Msun"]) * unyt.Msun

    masses = data.black_holes.subgrid_masses.to("Msun")
    accreted_masses = data.black_holes.total_accreted_masses.to("Msun")
    values = 1.0 - (accreted_masses + seed_mass) / masses
    values = values * unyt.dimensionless

    return masses, values


def make_single_image(
    filenames, names, mass_bounds, value_bounds, number_of_simulations, output_path
):

    fig, ax = plt.subplots()

    ax.set_xlabel(r"Black Hole Subgrid Masses $M_{\rm sub}$ [M$_\odot$]")
    ax.set_ylabel(r"Fraction of Mass Grown Through Mergers")
    ax.set_xscale("log")

    for filename, name in zip(filenames, names):
        masses, values = get_data(filename)
        mass_bins = (
            np.logspace(np.log10(mass_bounds[0]), np.log10(mass_bounds[1]), bins)
            * unyt.Msun
        )
        mass_bins = mass_bins.to("Msun")
        centers, medians, deviations, additional_x, additional_y = binned_median_line(
            masses, values, mass_bins, return_additional=True
        )

        fill_plot, = ax.plot(centers, medians, label=name)
        ax.fill_between(
            centers,
            medians - deviations[0],
            medians + deviations[1],
            alpha=0.2,
            facecolor=fill_plot.get_color(),
        )
        if number_of_simulations == 1 and plot_scatter:
            scatter_x_against_y(ax, masses, values)
        ax.scatter(additional_x.value, additional_y.value, color=fill_plot.get_color())

    ax.legend()
    ax.set_xlim(*mass_bounds)
    ax.set_ylim(*value_bounds)

    fig.savefig(f"{output_path}/black_hole_merger_mass_fractions.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(
        description="Merger Mass Fraction - BH mass relation"
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
