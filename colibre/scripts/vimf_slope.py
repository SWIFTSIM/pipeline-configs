"""
Plots the SNII energy fraction vs. stellar birth density.
"""

import matplotlib.pyplot as plt
import numpy as np
import unyt

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

def get_snapshot_param_string(data, param_name: str) -> float:
    try:
        return str(data.metadata.parameters[param_name].decode("utf-8"))
    except KeyError:
        raise KeyError(f"Parameter {param_name} not found in snapshot metadata.")


arguments = ScriptArgumentParser(
    description="Creates a plot of vIMF high mass slope vs. IMF variable"
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

ax_variables = []

# Begin plotting
fig, ax = plt.subplots(1, 1)

for color, (snapshot, name) in enumerate(zip(data, names)):

    try:
        variable_name = get_snapshot_param_string(
            snapshot, "COLIBREFeedback:IMF_Scaling_Variable"
        )
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
        pivot = get_snapshot_param_float(
            snapshot, "COLIBREFeedback:IMF_sigmoid_pivot_CGS"
        )

        if variable_name == 'Density':
            variable_bounds = [1e-2, 1e5]
            dim = "1/cm**3"
            xlabel = "Stellar Birth Density $\u03c1_B/m_H$ [cm$^{-3}$]"
        
        elif variable_name == 'Pressure':
            variable_bounds = [1e2, 1e9]
            dim = "K/cm**3"
            xlabel = "Stellar Birth Pressure $P_B/k$ [K cm$^{-3}$]"
        
        elif variable_name == 'Redshift':
            variable_bounds = [4, 20]
            dim = "dimensionless"
            xlabel = "Birth Redshift"

        log_variable_bounds = np.log10(variable_bounds)

        imf_variable = unyt.unyt_array(
            np.logspace(*log_variable_bounds, number_of_bins), dim
        )

        # Compute energy fractions
        slope_values = variable_slope(
            density=imf_variable,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            pivot_density=pivot,
            sigma=sigma,
        )

        if variable_name not in ax_variables:
            if len(ax_variables) == 0:
                axis = ax
            else:
                ax2 = ax.twiny()
                axis = ax2
            axis.set_xlim(*10**log_variable_bounds)
            axis.set_xscale("log")
            axis.set_xlabel(xlabel)

            if variable_name == 'Redshift':
                t_ticks = np.array([4.0, 6.0, 8.0, 10.0, 12.0, 20])
                axis.set_xticks(t_ticks)
                axis.set_xticklabels([f"${int(z)}$" for z in t_ticks])

        else:
            if len(ax_variables) > 0 and ax_variables[0] != variable_name:
                axis = ax2
            else:
                axis = ax

        ax_variables.append(variable_name)
        

    except KeyError as e:
        print(e)
        # Default to -2.3 (Chabrier IMF) if any parameter is missing
        imf_variable = unyt.unyt_array(
            np.logspace(-3, 9, number_of_bins), "dimensionless"
        )
        slope_values = unyt.unyt_array(
            -2.3*np.ones(number_of_bins), "dimensionless"
        )
        pivot = 1e-10
        axis = ax

    z = snapshot.metadata.z
    axis.plot(
        imf_variable.value,
        slope_values,
        color=f"C{color}",
    )

    ax.plot(
        [],
        label=f"{name} ($z={z:.1f})$",
        color=f"C{color}",
        )

    # Add the pivot density line
    axis.axvline(
        pivot, color=f"C{color}", linestyle="dashed", zorder=-10, alpha=0.5
    )

ax.legend(loc="upper left", markerfirst=False)

fig.savefig(f"{arguments.output_directory}/vimf_top_heavy_slope.png")
