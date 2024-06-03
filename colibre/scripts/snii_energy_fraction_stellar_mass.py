"""
Plots the SNII energy fraction at galaxy median stellar
birth pressure vs. galaxy stellar mass.
"""


import matplotlib.pyplot as plt
import numpy as np
import unyt

from swiftpipeline.argumentparser import ScriptArgumentParser
from velociraptor import load as load_catalogue
from swiftsimio import load as load_snapshot
from scipy import stats

# Set x-axis limits
mass_bounds = [1e5, 1e13]  # Msun
log_mass_bounds = np.log10(mass_bounds)  # Msun


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
    description="the SNII energy fraction at galaxy median stellar birth pressure vs. galaxy stellar mass."
)

snapshot_filenames = [
    f"{directory}/{snapshot}"
    for directory, snapshot in zip(arguments.directory_list, arguments.snapshot_list)
]
catalogue_filenames = [
    f"{directory}/{catalogue}"
    for directory, catalogue in zip(arguments.directory_list, arguments.catalogue_list)
]

plt.style.use(arguments.stylesheet_location)

aperture_size = 50  # kpc

number_of_bins = 41
stellar_mass_bins = np.linspace(*log_mass_bounds, number_of_bins)
stellar_mass_centers = 0.5 * (stellar_mass_bins[1:] + stellar_mass_bins[:-1])

# Creating plot
fig, ax = plt.subplots()
ax.set_xlabel(f"ExclusiveSphere {aperture_size}kpc StellarMass $\\rm [M_\odot]$")
ax.set_ylabel("SNII Energy Fraction $f_{\\rm E}$ at Median Birth Pressure")
ax.set_xscale("log")

for color, (snp_filename, cat_filename, name) in enumerate(
    zip(snapshot_filenames, catalogue_filenames, arguments.name_list)
):
    catalogue = load_catalogue(cat_filename)
    snapshot = load_snapshot(snp_filename)
    z = snapshot.metadata.z

    star_mass = catalogue.get_quantity(f"apertures.mass_star_{aperture_size}_kpc")
    mask = star_mass > mass_bounds[0] * star_mass.units
    star_mass = star_mass[mask]
    birth_pressures = catalogue.get_quantity(
        f"fofsubhaloproperties.medianstellarbirthpressure"
    ).to("K/cm**3")[mask]

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

    # Compute binned statistics
    log_star_mass = np.log10(star_mass)
    bin_stats = [
        stats.binned_statistic(
            log_star_mass, energy_fractions, bins=stellar_mass_bins, statistic=stat
        )
        for stat in [
            "median",
            lambda x: np.percentile(x, 16),
            lambda x: np.percentile(x, 84),
        ]
    ]
    energy_fractions_median, energy_fractions_lower, energy_fractions_upper = (
        stat[0] for stat in bin_stats
    )

    ax.plot(
        10 ** stellar_mass_centers,
        energy_fractions_median,
        label=f"{name} ($z={z:.1f})$",
        color=f"C{color}",
    )
    ax.fill_between(
        10 ** stellar_mass_centers,
        energy_fractions_upper,
        energy_fractions_lower,
        color=f"C{color}",
        alpha=0.3,
        zorder=-10,
    )

ax.legend()
ax.set_xlim(*mass_bounds)

fig.savefig(f"{arguments.output_directory}/snii_energy_fraction_stellar_mass.png")
