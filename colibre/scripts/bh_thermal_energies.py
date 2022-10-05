import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load

from velociraptor.tools.lines import binned_median_line
from velociraptor.autoplotter.plot import scatter_x_against_y
import unyt

# Set the limits of the figure.
mass_bounds = [1e3, 3e10]  # Msun
value_bounds = [1e55, 1e63]  # erg
bins = 25
plot_scatter = True


def get_data(filename):

    data = load(filename)

    masses = data.black_holes.subgrid_masses.to("Msun")
    values = data.black_holes.agntotal_injected_energies.astype(np.float64).to("erg")

    return masses, values


def make_single_image(
    filenames, names, mass_bounds, value_bounds, number_of_simulations, output_path
):

    fig, ax = plt.subplots()

    ax.set_xlabel("Black Hole Subgrid Masses $M_{\\rm sub}$ [M$_\odot$]")
    ax.set_ylabel(
        "Black Hole Injected Thermal AGN Energies $E_{\\rm th}$ $[{\\rm erg}]$"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")

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

    fig.savefig(f"{output_path}/black_hole_thermal_energies.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(
        description="AGN injected thermal energies - BH mass relation"
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
