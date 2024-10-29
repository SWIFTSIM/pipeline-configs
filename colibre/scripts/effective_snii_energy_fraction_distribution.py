"""
Plots the effective SNII energy fraction distribution.
"""

import matplotlib.pyplot as plt
import numpy as np
import unyt

from unyt import mh

from scipy.integrate import quad

from swiftsimio import load
from swiftpipeline.argumentparser import ScriptArgumentParser


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

def imf_func(
        mass: float,
        imf_type: str,
        slope_low: float,
        slope_high: float,
        mass_weighted: bool):
    """
    Computes the (unnormalised) IMF value at a given star mass.

    Parameters:
    mass (float): star mass
    imf_type (str): specifies if IMF variations are bottom-heavy or top-heavy or the IMF is universal (Chabrier etc.)
    slope_low (float): IMF slope value m < 0.5 Msolar
    slope_high (float): IMF slope value m > 0.5 Msolar
    mass_weighted (bool): If True, returns m*dN/dm to normalise the IMF

    Returns:
    float: The IMF value dN/dM
    """
    if imf_type == 'Chabrier':
        if mass > 1:
            imf = 0.237912 * pow(mass, -2.3)
        else:
            log_mass = np.log10(mass)
            imf = 0.852464 * np.exp((log_mass - np.log10(0.079)) * (log_mass - np.log10(0.079)) / (-2.0 * 0.69 * 0.69)) / mass

    else: #broken power-law
        if imf_type == 'Kroupa':
            slope_low, slope_high = -1.3, -2.3
        
        if mass < 0.5:
            imf = 0.5**(slope_high-slope_low) * pow(mass,slope_low)
        else:
            imf = pow(mass,slope_high)

    if mass_weighted == True:
        imf *= mass
             
    return imf

def N_cc_func(
        imf_type: str,
        slope_low: float,
        slope_high: float):
    """
    Computes the number of core-collapse SNe for given IMF slope values using imf_func

    Parameters:
    imf_type (str): specifies if IMF variations are bottom-heavy or top-heavy or the IMF is universal (Chabrier etc.)
    slope_low (float): Minimum IMF slope value
    slope_high (float): Maximum IMF slope value

    Returns:
    float: Number of core-collapse SNe events for a stellar population of initial mass 1 Msolar
    """
    imf_lower_limit = 0.1
    imf_upper_limit = 100
    cc_threshold = 8 # minimum star mass [Msolar] for core-collapse SNe

    N_cc = quad(imf_func,cc_threshold,imf_upper_limit,args=(imf_type,slope_low,slope_high,False))[0]
    normalise_vimf = quad(imf_func,imf_lower_limit,imf_upper_limit,args=(imf_type,slope_low,slope_high,True))[0]
    N_cc /= normalise_vimf

    return N_cc



def get_snapshot_param_float(snapshot, param_name: str) -> float:
    try:
        return float(snapshot.metadata.parameters[param_name].decode("utf-8"))
    except KeyError:
        raise KeyError(f"Parameter {param_name} not found in snapshot metadata.")


arguments = ScriptArgumentParser(
    description="Creates an SNII energy fraction distribution plot, split by redshift"
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

energy_fraction_bins = unyt.unyt_array(
    np.linspace(0, 5.0, number_of_bins), units="dimensionless"
)
energy_fraction_bin_width = (
    energy_fraction_bins[1].value - energy_fraction_bins[0].value
)
energy_fraction_centers = 0.5 * (energy_fraction_bins[1:] + energy_fraction_bins[:-1])


# Begin plotting
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
axes = axes.flat

ax_dict = {"$z < 1$": axes[0], "$1 < z < 3$": axes[1], "$z > 3$": axes[2]}

for label, ax in ax_dict.items():
    ax.text(0.025, 1.0 - 0.025 * 3, label, transform=ax.transAxes, ha="left", va="top")

for color, (snapshot, name) in enumerate(zip(data, names)):

    birth_densities = snapshot.stars.birth_densities.to("g/cm**3") / mh.to("g")
    birth_temperatures = snapshot.stars.birth_temperatures.to("K")
    birth_pressures = (birth_densities * birth_temperatures).to("K/cm**3")
    birth_redshifts = 1 / snapshot.stars.birth_scale_factors.value - 1

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

        # Compute Ncc per particle, need to compute slope values per particle first


        # effective_energy_fractions = N_cc/ N_cc [Chabrier] * energy_fractions


    except KeyError as e:
        print(e)
        # Default to -1 if any parameter is missing
        energy_fractions = unyt.unyt_array(
            -np.ones_like(birth_pressures.value), "dimensionless"
        )

    # Segment birth pressures into redshift bins
    energy_fraction_by_redshift = {
        "$z < 5$": energy_fractions[birth_redshifts < 5],
        "$5 < z < 7$": energy_fractions[
            np.logical_and(birth_redshifts > 5, birth_redshifts < 7)
        ],
        "$7 < z < 10$": energy_fractions[
            np.logical_and(birth_redshifts > 7, birth_redshifts < 10)
        ],
        "$z > 10$": energy_fractions[birth_redshifts > 10],
    }


    # Total number of stars formed
    Num_of_stars_total = len(birth_redshifts)

    # Average energy fraction (computed among all star particles)
    average_energy_fraction = np.mean(energy_fractions)

    for redshift, ax in ax_dict.items():
        data = energy_fraction_by_redshift[redshift]

        H, _ = np.histogram(data, bins=energy_fraction_bins)
        y_points = H / energy_fraction_bin_width / Num_of_stars_total

        ax.plot(energy_fraction_centers, y_points, label=name, color=f"C{color}")

        # Add the median snii energy fraction
        ax.axvline(
            np.median(data),
            color=f"C{color}",
            linestyle="dashed",
            zorder=-10,
            alpha=0.5,
        )

        ax.axvline(
            average_energy_fraction,
            color=f"C{color}",
            linestyle="dotted",
            zorder=-10,
            alpha=0.5,
        )

axes[0].legend(loc="upper right", markerfirst=False)
axes[2].set_xlabel("Effective SNII Energy Fraction $f'_{\\rm E}$")
axes[1].set_ylabel("$N_{\\rm bin}$ / d$f_{\\rm E}$ / $N_{\\rm total}$")

fig.savefig(f"{arguments.output_directory}/effective_snii_energy_fraction_distribution.png")
