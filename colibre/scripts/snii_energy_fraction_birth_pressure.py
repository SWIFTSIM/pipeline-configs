"""
Plots the SNII energy fraction vs. stellar birth pressure.
"""

import matplotlib.pyplot as plt
import numpy as np
import unyt

from swiftsimio import load
from swiftpipeline.argumentparser import ScriptArgumentParser

# Set x-axis limits
pressure_bounds = [1e1, 1e9]
log_pressure_bounds = np.log10(pressure_bounds)


def energy_fraction(
    pressure: unyt.unyt_array,
    fmin: float,
    fmax: float,
    sigma: float,
    pivot_pressure: float,
) -> unyt.unyt_array:
    """
    Computes the energy fraction in SNII feedback vs. stellar birth pressure.

    Parameters:
    pressure (unyt.unyt_array): The stellar birth pressure.
    fmin (float): The minimum energy fraction.
    fmax (float): The maximum energy fraction.
    sigma (float): The dispersion parameter.
    pivot_pressure (float): The pivot pressure.

    Returns:
    unyt.unyt_array: The computed energy fraction as a dimensionless unyt array.
    """

    slope = -1.0 / np.log(10) / sigma
    f_E = fmin + (fmax - fmin) / (1.0 + (pressure.value / pivot_pressure) ** slope)
    return unyt.unyt_array(f_E, "dimensionless")


def get_snapshot_param_float(snapshot, param_name: str) -> float:
    try:
        return float(snapshot.metadata.parameters[param_name].decode("utf-8"))
    except KeyError:
        raise KeyError(f"Parameter {param_name} not found in snapshot metadata.")


arguments = ScriptArgumentParser(
    description="Creates a plot of SNII energy fraction vs. birth pressure"
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
birth_pressures = unyt.unyt_array(
    np.logspace(*log_pressure_bounds, number_of_bins), "K/cm**3"
)

# Begin plotting
fig, ax = plt.subplots(1, 1)
for color, (snapshot, name) in enumerate(zip(data, names)):

    # Default value that will be plotted outside the plot's domain
    # if the true value is not found in the snapshot's metadata
    pivot_pressure = 1e-10

    try:
        # Extract feedback parameters from snapshot metadata
        fmin = get_snapshot_param_float(
            snapshot, "COLIBREFeedback:SNII_energy_fraction_min"
        )
        fmax = get_snapshot_param_float(
            snapshot, "COLIBREFeedback:SNII_energy_fraction_max"
        )
        sigma = get_snapshot_param_float(
            snapshot, "COLIBREFeedback:SNII_energy_fraction_sigma_P"
        )
        pivot_pressure = get_snapshot_param_float(
            snapshot, "COLIBREFeedback:SNII_energy_fraction_P_0_K_p_cm3"
        )

        # Compute energy fractions
        energy_fractions = energy_fraction(
            pressure=birth_pressures,
            fmin=fmin,
            fmax=fmax,
            pivot_pressure=pivot_pressure,
            sigma=sigma,
        )
    except KeyError as e:
        print(e)
        # Default to -1 if any parameter is missing
        energy_fractions = unyt.unyt_array(
            -np.ones_like(birth_pressures.value), "dimensionless"
        )

    z = snapshot.metadata.z
    ax.plot(
        birth_pressures.value,
        energy_fractions,
        label=f"{name} ($z={z:.1f})$",
        color=f"C{color}",
    )

    # Add the pivot pressure line
    ax.axvline(
        pivot_pressure, color=f"C{color}", linestyle="dashed", zorder=-10, alpha=0.5
    )

ax.set_xlim(*pressure_bounds)
ax.set_xscale("log")
ax.legend(loc="upper left", markerfirst=False)
ax.set_ylabel("SNII Energy Fraction $f_{\\rm E}$")
ax.set_xlabel("Stellar Birth Pressure $P_B/k$ [K cm$^{-3}$]")

fig.savefig(f"{arguments.output_directory}/snii_energy_fraction_birth_pressure.png")
