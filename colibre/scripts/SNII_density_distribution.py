"""
Plots the SNII density distribution.
"""

import matplotlib.pyplot as plt
import numpy as np
import unyt

from unyt import mh, cm

from swiftsimio import load

try:
    plt.style.use("mnras.mplstyle")
except:
    pass

from swiftpipeline.argumentparser import ScriptArgumentParser

arguments = ScriptArgumentParser(
    description="Creates a plot showing the distribution of the gas densities recorded when the gas was last heated by SNII, split by redshift"
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

SNII_density_bins = unyt.unyt_array(
    np.logspace(-5, 5, number_of_bins), units=1 / cm ** 3
)
log_SNII_density_bin_width = np.log10(SNII_density_bins[1].value) - np.log10(
    SNII_density_bins[0].value
)
SNII_density_centers = 0.5 * (SNII_density_bins[1:] + SNII_density_bins[:-1])


# Begin plotting

fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
axes = axes.flat

ax_dict = {
    "$z < 1$": axes[0],
    "$1 < z < 3$": axes[1],
    "$z > 3$": axes[2],
}

for label, ax in ax_dict.items():
    ax.loglog()
    ax.text(0.025, 1.0 - 0.025 * 3, label, transform=ax.transAxes, ha="left", va="top")

for color, (snapshot, name) in enumerate(zip(data, names)):

    stars_SNII_densities = (snapshot.stars.densities_at_last_supernova_event / mh).to(
        SNII_density_bins.units
    )
    stars_SNII_redshifts = 1 / snapshot.stars.last_sniifeedback_scale_factors.value - 1

    gas_SNII_densities = (snapshot.gas.densities_at_last_supernova_event / mh).to(
        SNII_density_bins.units
    )
    gas_SNII_redshifts = 1 / snapshot.gas.last_sniifeedback_scale_factors.value - 1

    # Segment SNII densities into redshift bins
    stars_SNII_densities_by_redshift = {
        "$z < 1$": stars_SNII_densities[stars_SNII_redshifts < 1],
        "$1 < z < 3$": stars_SNII_densities[
            np.logical_and(stars_SNII_redshifts > 1, stars_SNII_redshifts < 3)
        ],
        "$z > 3$": stars_SNII_densities[stars_SNII_redshifts > 3],
    }

    gas_SNII_densities_by_redshift = {
        "$z < 1$": gas_SNII_densities[gas_SNII_redshifts < 1],
        "$1 < z < 3$": gas_SNII_densities[
            np.logical_and(gas_SNII_redshifts > 1, gas_SNII_redshifts < 3)
        ],
        "$z > 3$": gas_SNII_densities[gas_SNII_redshifts > 3],
    }

    for redshift, ax in ax_dict.items():
        data = np.concatenate(
            [
                stars_SNII_densities_by_redshift[redshift],
                gas_SNII_densities_by_redshift[redshift],
            ]
        )

        H, _ = np.histogram(data, bins=SNII_density_bins)

        # Total number SNII-heated gas particles
        Num_of_obj = np.sum(H)

        # Check to avoid division by zero
        if Num_of_obj:
            y_points = H / log_SNII_density_bin_width / Num_of_obj
        else:
            y_points = np.zeros_like(H)

        ax.plot(
            SNII_density_centers, y_points, label=name, color=f"C{color}",
        )
        ax.axvline(
            np.median(data),
            color=f"C{color}",
            linestyle="dashed",
            zorder=-10,
            alpha=0.5,
        )


axes[0].legend(loc="upper right", markerfirst=False)
axes[2].set_xlabel(
    "Density of the gas heated by SNII $\\rho_{\\rm SNII}$ [$n_H$ cm$^{-3}$]"
)
axes[1].set_ylabel("$N_{\\rm bin}$ / d$\\log\\rho_{\\rm SNII}$ / $N_{\\rm total}$")

fig.savefig(f"{arguments.output_directory}/SNII_density_distribution.png")
