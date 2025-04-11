"""
Plots the birth density distribution.
"""

import matplotlib.pyplot as plt
import numpy as np
import unyt
import traceback

from unyt import mh

from swiftsimio import load
from swiftpipeline.argumentparser import ScriptArgumentParser

def variable_slope(
    density: unyt.unyt_array,
    alpha_min: float,
    alpha_max: float,
    sigma: float,
    pivot_density: float,
) -> unyt.unyt_array:
    """
    Computes the IMF high mass slope value at a given density.

    Parameters:
    density (unyt.unyt_array): The stellar birth density.
    alpha_min (float): The minimum slope value.
    alpha_max (float): The maximum slope value.
    sigma (float): The dispersion parameter.
    pivot_density (float): The pivot density.

    Returns:
    unyt.unyt_array: The computed slope value as a dimensionless unyt array.
    """

    alpha = (alpha_min - alpha_max) / (1.0 + np.exp(sigma * np.log10( density.value / pivot_density )) ) + alpha_max
    return unyt.unyt_array(alpha, "dimensionless")

def get_snapshot_param_float(snapshot, param_name: str) -> float:
    try:
        return float(snapshot.metadata.parameters[param_name].decode("utf-8"))
    except KeyError:
        raise KeyError(f"Parameter {param_name} not found in snapshot metadata.")

arguments = ScriptArgumentParser(
    description="Creates a stellar birth density distribution plot, split high mass IMF slope values"
)

snapshot_filenames = [
    f"{directory}/{snapshot}"
    for directory, snapshot in zip(arguments.directory_list, arguments.snapshot_list)
]

names = arguments.name_list
output_path = arguments.output_directory

plt.style.use(arguments.stylesheet_location)

data = [load(snapshot_filename) for snapshot_filename in snapshot_filenames]

number_of_bins = 256

birth_density_bins = unyt.unyt_array(
    np.logspace(-2, 7, number_of_bins), units="1/cm**3"
)
log_birth_density_bin_width = np.log10(birth_density_bins[1].value) - np.log10(
    birth_density_bins[0].value
)
birth_density_centers = 0.5 * (birth_density_bins[1:] + birth_density_bins[:-1])


# Begin plotting
fig, axes = plt.subplots(4, 1, sharex=True, sharey=True)
axes = axes.flat

z = data[0].metadata.z

ax_dict = {"$\u03b1 < -2.25 $": axes[0], "$-2.25 < \u03b1 < -1.95 $": axes[1], "$-1.95 < \u03b1 < -1.75 $": axes[2], "$-1.75 < \u03b1$": axes[3]}


for label, ax in ax_dict.items():
    ax.loglog()
    ax.text(0.025, 1.0 - 0.025 * 3, label, transform=ax.transAxes, ha="left", va="top")

for color, (snapshot, name) in enumerate(zip(data, names)):

    birth_densities = snapshot.stars.birth_densities.to("g/cm**3") / mh.to("g")
    birth_redshifts = 1 / snapshot.stars.birth_scale_factors.value - 1

    try:

        # Extract IMF parameters from snapshot metadata
        alpha_min = get_snapshot_param_float(
            snapshot, "COLIBREFeedback:IMF_HighMass_slope_minimum"
        )
        alpha_max = get_snapshot_param_float(
            snapshot, "COLIBREFeedback:IMF_HighMass_slope_maximum"
        )
        sigma = get_snapshot_param_float(
            snapshot, "COLIBREFeedback:IMF_sigmoid_inverse_width"
        )
        pivot_density = get_snapshot_param_float(
            snapshot, "COLIBREFeedback:IMF_sigmoid_pivot_CGS"
        )

        # Compute slope values
        slope_values = variable_slope(
            density=birth_densities,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            sigma=sigma,
            pivot_density=pivot_density,
        )

  
    except KeyError as e:
        print(e)
        slope_values = -2.3 * unyt.unyt_array(
            np.ones_like(birth_densities.value), "dimensionless"
        )


    # Segment birth densities into IMF slope bins

    birth_density_by_slope = {
        "$\u03b1 < -2.25 $": birth_densities[slope_values < 2.25],
        "$-2.25 < \u03b1 < -1.95 $": birth_densities[
            np.logical_and(slope_values > -2.25, slope_values < -1.95)
        ],
        "$-1.95 < \u03b1 < -1.75 $": birth_densities[
            np.logical_and(slope_values > -1.95, slope_values < -1.75)
        ],
        "$-1.75 < \u03b1$": birth_densities[slope_values > -1.75],
    }

    # Total number of stars formed
    Num_of_stars_total = len(birth_densities)

    for redshift, ax in ax_dict.items():
        data = birth_density_by_slope[redshift]
        if data.shape[0] == 0:
            continue
        if data.shape[0] == 0:
            continue

        H, _ = np.histogram(data, bins=birth_density_bins)
        y_points = H / log_birth_density_bin_width / Num_of_stars_total

        ax.plot(birth_density_centers, y_points, label=name, color=f"C{color}")

        # Add the median stellar birth-density line
        ax.axvline(
            np.median(data),
            color=f"C{color}",
            linestyle="dashed",
            zorder=-10,
            alpha=0.5,
        )


axes[0].legend(loc="upper right", markerfirst=False)
axes[3].set_xlabel("Stellar Birth Density $\\rho_B$ [$n_H$ cm$^{-3}$]")
axes[1].set_ylabel("$N_{\\rm bin}$ / d$\\log\\rho_B$ / $N_{\\rm total}$")

fig.savefig(f"{arguments.output_directory}/birth_density_distribution_alpha_binned.png")
