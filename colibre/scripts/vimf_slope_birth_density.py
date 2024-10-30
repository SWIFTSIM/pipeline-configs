"""
Plots the SNII energy fraction vs. stellar birth density.
"""

import matplotlib.pyplot as plt
import numpy as np
import unyt

from swiftsimio import load
from swiftpipeline.argumentparser import ScriptArgumentParser

# Set x-axis limits
density_bounds = [1e-2, 1e5]
log_density_bounds = np.log10(density_bounds)


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
    description="Creates a plot of vIMF high mass slope vs. birth density"
)


snapshot_filenames = [
    f"{directory}/{snapshot}"
    for directory, snapshot in zip(arguments.directory_list, arguments.snapshot_list)
]

names = arguments.name_list
output_path = arguments.output_directory

plt.style.use(arguments.stylesheet_location)

data = [load(snapshot_filename) for snapshot_filename in snapshot_filenames]

number_of_bins = 128
birth_densities = unyt.unyt_array(
    np.logspace(*log_density_bounds, number_of_bins), "1/cm**3"
)

# Begin plotting
fig, ax = plt.subplots(1, 1)
for color, (snapshot, name) in enumerate(zip(data, names)):

    # Default value that will be plotted outside the plot's domain
    # if the true value is not found in the snapshot's metadata
    pivot_density = 1e-10

    try:
        # Extract feedback parameters from snapshot metadata
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

        # Compute energy fractions
        slope_values = variable_slope(
            density=birth_densities,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            pivot_density=pivot_density,
            sigma=sigma,
        )

    except KeyError as e:
        print(e)
        # Default to -2.3 (Chabrier IMF) if any parameter is missing
        slope_values = unyt.unyt_array(
            -2.3*np.ones_like(birth_densities.value), "dimensionless"
        )

    z = snapshot.metadata.z
    ax.plot(
        birth_densities.value,
        slope_values,
        label=f"{name} ($z={z:.1f})$",
        color=f"C{color}",
    )

    # Add the pivot density line
    ax.axvline(
        pivot_density, color=f"C{color}", linestyle="dashed", zorder=-10, alpha=0.5
    )

ax.set_xlim(*density_bounds)
ax.set_xscale("log")
ax.legend(loc="upper left", markerfirst=False)
ax.set_ylabel("IMF high mass slope")
ax.set_xlabel("Stellar Birth density $\u03c1_B/m_H$ [cm$^{-3}$]")

fig.savefig(f"{arguments.output_directory}/vimf_top_heavy_slope.png")
