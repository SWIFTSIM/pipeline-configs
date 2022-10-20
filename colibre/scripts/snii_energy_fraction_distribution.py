"""
Plots the SNII energy fraction distribution.
"""

import matplotlib.pyplot as plt
import numpy as np
import unyt

from unyt import mh

from swiftsimio import load
from swiftpipeline.argumentparser import ScriptArgumentParser


def energy_fraction(
    pressure: unyt.unit_object,
    fmin: float,
    fmax: float,
    sigma: float,
    pivot_pressure: float,
) -> unyt.unit_object:
    """
    Computes the energy fractions in SNII feedback based on stellar birth pressure.
    """
    slope = -1.0 / np.log(10) / sigma
    f_E = fmin + (fmax - fmin) / (1.0 + (pressure.value / pivot_pressure) ** slope)
    return unyt.unyt_array(f_E, "dimensionless")


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
        fmin = float(
            snapshot.metadata.parameters[
                "COLIBREFeedback:SNII_energy_fraction_min"
            ].decode("utf-8")
        )
        fmax = float(
            snapshot.metadata.parameters[
                "COLIBREFeedback:SNII_energy_fraction_max"
            ].decode("utf-8")
        )
        sigma = float(
            snapshot.metadata.parameters[
                "COLIBREFeedback:SNII_energy_fraction_sigma_P"
            ].decode("utf-8")
        )
        pivot_pressure = float(
            snapshot.metadata.parameters[
                "COLIBREFeedback:SNII_energy_fraction_P_0_K_p_cm3"
            ].decode("utf-8")
        )

        energy_fractions = energy_fraction(
            pressure=birth_pressures,
            fmin=fmin,
            fmax=fmax,
            pivot_pressure=pivot_pressure,
            sigma=sigma,
        )
    except KeyError:
        energy_fractions = unyt.unyt_array(
            -np.ones_like(birth_pressures.value), "dimensionless"
        )

    # Segment birth pressures into redshift bins
    energy_fraction_by_redshift = {
        "$z < 1$": energy_fractions[birth_redshifts < 1],
        "$1 < z < 3$": energy_fractions[
            np.logical_and(birth_redshifts > 1, birth_redshifts < 3)
        ],
        "$z > 3$": energy_fractions[birth_redshifts > 3],
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
axes[2].set_xlabel("SNII Energy Fraction $f_{\\rm E}$")
axes[1].set_ylabel("$N_{\\rm bin}$ / d$f_{\\rm E}$ / $N_{\\rm total}$")

fig.savefig(f"{arguments.output_directory}/snii_energy_fraction_distribution.png")
