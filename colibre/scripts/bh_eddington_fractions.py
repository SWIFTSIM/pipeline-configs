import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load
from velociraptor.autoplotter.objects import VelociraptorLine
from velociraptor.tools.lines import binned_median_line
from velociraptor.autoplotter.plot import scatter_x_against_y

import unyt
from unyt import G, c, sigma_thompson, mp

# Set the limits of the figure.
mass_bounds = [1e3, 3e10]  # Msun
value_bounds = [1e-7, 1e1]  # dimensionless
bins = 25
plot_scatter = True


def compute_eddington_fractions(
    acc_rates: unyt.unyt_array,
    bh_masses: unyt.unyt_array,
    radiative_efficiencies: unyt.unyt_array,
) -> unyt.unyt_array:

    eddington_rates = (
        4.0 * np.pi * G * bh_masses * mp / (radiative_efficiencies * c * sigma_thompson)
    )
    eddington_fractions = acc_rates / eddington_rates
    return eddington_fractions


def get_data(filename):

    data = load(filename)

    masses = data.black_holes.subgrid_masses.to("Msun")

    try:
        values = data.black_holes.eddington_fractions

    # Compute Eddington fractions yourself if they are not stored in the snapshot
    except AttributeError:
        acc_rates = data.black_holes.accretion_rates

        # Jet model
        try:
            rad_effs = data.black_holes.radiative_efficiencies
        # Purely thermal AGN feedback
        except AttributeError:
            rad_effs = unyt.unyt_array(
                float(
                    data.metadata.parameters["COLIBREAGN:radiative_efficiency"].decode(
                        "utf-8"
                    )
                )
                * np.ones_like(masses),
                "dimensionless",
            )

        values = compute_eddington_fractions(
            acc_rates=acc_rates, bh_masses=masses, radiative_efficiencies=rad_effs
        )

    return masses, values


def make_single_image(
    filenames, names, mass_bounds, value_bounds, number_of_simulations, output_path
):

    fig, ax = plt.subplots()

    ax.set_xlabel("Black Hole Subgrid Masses $M_{\\rm sub}$ [M$_\\odot$]")
    ax.set_ylabel(
        r"Black Hole Eddington Fraction $\dot{m}=\dot{M}_{\rm BH}/\dot{M}_{\rm Edd}$"
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

    ax.plot(
        [1e-10, 1e11],
        [0.01, 0.01],
        color="black",
        linestyle=":",
        linewidth=0.75,
        label=r"$\dot{m}=0.01$",
    )

    ax.legend()
    ax.set_xlim(*mass_bounds)
    ax.set_ylim(*value_bounds)

    fig.savefig(f"{output_path}/black_hole_eddington_fractions.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(
        description="Eddington fraction - BH mass relation"
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
