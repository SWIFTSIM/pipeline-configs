
"""
Creates the plot of metallicity against birth density, with
the background coloured by f_E.
"""

import matplotlib.pyplot as plt
import numpy as np
import unyt

from swiftsimio import load

from unyt import mh, cm, Gyr
from matplotlib.colors import LogNorm, Normalize

# Constants; these could be put in the parameter file but are rarely changed.
number_of_bins = 128

birth_density_bins = unyt.unyt_array(
    np.logspace(-3, 5, number_of_bins), units=1 / cm ** 3
)
metal_mass_fraction_bins = unyt.unyt_array(
    np.logspace(-6, 0, number_of_bins), units="dimensionless"
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

    fig, ax = plt.subplots(
        vertical_number, horizontal_number, squeeze=True, sharex=True, sharey=True,
    )

    ax = np.array([ax]) if number_of_simulations == 1 else ax

    if horizontal_number * vertical_number > number_of_simulations:
        for axis in ax.flat[number_of_simulations:]:
            axis.axis("off")

    # Set all valid on bottom row to have the horizontal axis label.
    for axis in np.atleast_2d(ax)[:][-1]:
        axis.set_xlabel("Birth Density [$n_H$ cm$^{-3}$]")

    for axis in np.atleast_2d(ax).T[:][0]:
        axis.set_ylabel("Smoothed MMF $Z$ []")

    ax.flat[0].loglog()

    return fig, ax


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

    snapshots = [load(filename) for filename in snapshot_filenames]

    fig, axes = setup_axes(number_of_simulations=arguments.number_of_inputs)

    for snapshot, ax in zip(snapshots, axes.flat):
        used_parameters = snapshot.metadata.parameters

        parameters = {
            k: float(used_parameters[v])
            for k, v in {
                "f_E,min": "EAGLEFeedback:SNII_energy_fraction_min",
                "f_E,max": "EAGLEFeedback:SNII_energy_fraction_max",
                "n_Z": "EAGLEFeedback:SNII_energy_fraction_n_Z",
                "n_n": "EAGLEFeedback:SNII_energy_fraction_n_n",
                "Z_pivot": "EAGLEFeedback:SNII_energy_fraction_Z_0",
                "n_pivot": "EAGLEFeedback:SNII_energy_fraction_n_0_H_p_cm3",
            }.items()
        }
        star_formation_parameters = {
            k: float(used_parameters[v])
            for k, v in {
                "threshold_Z0": "EAGLEStarFormation:threshold_Z0",
                "threshold_n0": "EAGLEStarFormation:threshold_norm_H_p_cm3",
                "slope": "EAGLEStarFormation:threshold_slope",
            }.items()
        }

        # Now need to make background grid of f_E.
        birth_density_grid, metal_mass_fraction_grid = np.meshgrid(
            0.5 * (birth_density_bins.value[1:] + birth_density_bins.value[:-1]),
            0.5 * (metal_mass_fraction_bins.value[1:] + metal_mass_fraction_bins.value[:-1]),
        )

        f_E_grid = parameters["f_E,min"] + (parameters["f_E,max"] - parameters["f_E,min"]) / (
            1.0
            + (metal_mass_fraction_grid / parameters["Z_pivot"]) ** parameters["n_Z"]
            * (birth_density_grid / parameters["n_pivot"]) ** (-parameters["n_n"])
        )

        # Begin plotting

        ax.loglog()

        mappable = ax.pcolormesh(
            birth_density_bins.value,
            metal_mass_fraction_bins.value,
            f_E_grid,
            norm=LogNorm(1e-2, 1e1, clip=True),
        )

        try:
            metal_mass_fractions = snapshot.stars.smoothed_metal_mass_fractions.value
        except AttributeError:
            metal_mass_fractions = snapshot.stars.metal_mass_fractions.value

        H, _, _ = np.histogram2d(
            (snapshot.stars.birth_densities / mh).to(1 / cm ** 3).value,
            metal_mass_fractions,
            bins=[birth_density_bins.value, metal_mass_fraction_bins.value],
        )

        ax.contour(birth_density_grid, metal_mass_fraction_grid, H.T, levels=6, cmap="Pastel1")

        # Add line showing SF law
        try:
            sf_threshold_density = star_formation_parameters["threshold_n0"] * (
                metal_mass_fraction_bins.value / star_formation_parameters["threshold_Z0"]
            ) ** (star_formation_parameters["slope"])
            ax.plot(
                sf_threshold_density,
                metal_mass_fraction_bins,
                linestyle="dashed",
                label="SF threshold",
            )
        except:
            pass

    legend = axes.flat[0].legend(markerfirst=True, loc="lower left")
    plt.setp(legend.get_texts(), color="white")

    fig.colorbar(mappable, label="Feedback energy fraction $f_E$")

    try:
        fontsize=legend.get_texts()[0].get_fontsize()
    except:
        fontsize=6


    for ax, snapshot, name in zip(axes, snapshots, arguments.name_list):
        used_parameters = snapshot.metadata.parameters

        parameters = {
            k: float(used_parameters[v])
            for k, v in {
                "f_E,min": "EAGLEFeedback:SNII_energy_fraction_min",
                "f_E,max": "EAGLEFeedback:SNII_energy_fraction_max",
                "n_Z": "EAGLEFeedback:SNII_energy_fraction_n_Z",
                "n_n": "EAGLEFeedback:SNII_energy_fraction_n_n",
                "Z_pivot": "EAGLEFeedback:SNII_energy_fraction_Z_0",
                "n_pivot": "EAGLEFeedback:SNII_energy_fraction_n_0_H_p_cm3",
            }.items()
        }

        ax.text(
            0.975,
            0.025,
            "\n".join(
                [f"${k.replace('_', '_{') + '}'}$: ${v:.4g}$" for k, v in parameters.items()]
            ),
            color="white",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=fontsize,
        )

        ax.text(
            0.975,
            0.975,
            "Contour lines linearly spaced",
            color="white",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=fontsize,
        )

        ax.text(
            0.025,
            0.025,
            name,
            ha="left",
            va="bottom",
            transform=ax.transAxes,
            color="white"
        )

    fig.savefig(f"{arguments.output_directory}/birth_density_metallicity.png")