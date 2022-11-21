"""
Plots the birth velocity dispersion distribution.
"""

import matplotlib.pyplot as plt
import numpy as np
import unyt
import traceback


from swiftsimio import load
from swiftpipeline.argumentparser import ScriptArgumentParser


arguments = ScriptArgumentParser(
    description="Creates a stellar birth velocity dispersion distribution plot, split by redshift"
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

birth_velocity_dispersion_bins = unyt.unyt_array(
    np.logspace(0.25, 5.25, number_of_bins), units="km**2/s**2"
)
log_birth_velocity_dispersion_bin_width = np.log10(
    birth_velocity_dispersion_bins[1].value
) - np.log10(birth_velocity_dispersion_bins[0].value)
birth_velocity_dispersion_centers = 0.5 * (
    birth_velocity_dispersion_bins[1:] + birth_velocity_dispersion_bins[:-1]
)


# Begin plotting
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
axes = axes.flat

ax_dict = {"$z < 1$": axes[0], "$1 < z < 3$": axes[1], "$z > 3$": axes[2]}

for label, ax in ax_dict.items():
    ax.loglog()
    ax.text(0.025, 1.0 - 0.025 * 3, label, transform=ax.transAxes, ha="left", va="top")

for color, (snapshot, name) in enumerate(zip(data, names)):

    birth_velocity_dispersions = snapshot.stars.birth_velocity_dispersions.to(
        "km**2/s**2"
    )
    birth_redshifts = 1 / snapshot.stars.birth_scale_factors.value - 1

    # Segment birth velocity dispersions into redshift bins
    birth_velocity_dispersion_by_redshift = {
        "$z < 1$": birth_velocity_dispersions[birth_redshifts < 1],
        "$1 < z < 3$": birth_velocity_dispersions[
            np.logical_and(birth_redshifts > 1, birth_redshifts < 3)
        ],
        "$z > 3$": birth_velocity_dispersions[birth_redshifts > 3],
    }

    # Total number of stars formed
    Num_of_stars_total = len(birth_redshifts)

    for redshift, ax in ax_dict.items():
        data = birth_velocity_dispersion_by_redshift[redshift]

        H, _ = np.histogram(data, bins=birth_velocity_dispersion_bins)
        y_points = H / log_birth_velocity_dispersion_bin_width / Num_of_stars_total

        ax.plot(
            birth_velocity_dispersion_centers, y_points, label=name, color=f"C{color}"
        )

        # Add the median stellar birth-velocity dispersion line
        ax.axvline(
            np.median(data),
            color=f"C{color}",
            linestyle="dashed",
            zorder=-10,
            alpha=0.5,
        )

axes[0].legend(loc="upper right", markerfirst=False)
axes[2].set_xlabel(
    "Stellar Birth Velocity Dispersion $\\sigma{}_B^2$ [km$^2$ s$^{-2}$]"
)
axes[1].set_ylabel("$N_{\\rm bin}$ / d$\\log(\\sigma{}_B^2)$ / $N_{\\rm total}$")

fig.savefig(f"{arguments.output_directory}/birth_velocity_dispersion_distribution.png")
