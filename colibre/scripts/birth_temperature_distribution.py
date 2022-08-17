"""
Plots the birth pressure distribution.
"""

import matplotlib.pyplot as plt
import numpy as np
import unyt
import traceback

from unyt import mh

from swiftsimio import load
from swiftpipeline.argumentparser import ScriptArgumentParser


arguments = ScriptArgumentParser(
    description="Creates a stellar birth temperature distribution plot, split by redshift"
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

birth_temperature_bins = unyt.unyt_array(
    np.logspace(1.0, 4.1, number_of_bins), units="K"
)
log_birth_temperature_bin_width = np.log10(birth_temperature_bins[1].value) - np.log10(
    birth_temperature_bins[0].value
)
birth_temperature_centers = 0.5 * (
    birth_temperature_bins[1:] + birth_temperature_bins[:-1]
)


# Begin plotting
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
axes = axes.flat

ax_dict = {"$z < 1$": axes[0], "$1 < z < 3$": axes[1], "$z > 3$": axes[2]}

for label, ax in ax_dict.items():
    ax.loglog()
    ax.text(0.025, 1.0 - 0.025 * 3, label, transform=ax.transAxes, ha="left", va="top")

for color, (snapshot, name) in enumerate(zip(data, names)):

    birth_temperatures = snapshot.stars.birth_temperatures.to("K")
    birth_redshifts = 1 / snapshot.stars.birth_scale_factors.value - 1

    # Segment birth pressures into redshift bins
    birth_temperature_by_redshift = {
        "$z < 1$": birth_temperatures[birth_redshifts < 1],
        "$1 < z < 3$": birth_temperatures[
            np.logical_and(birth_redshifts > 1, birth_redshifts < 3)
        ],
        "$z > 3$": birth_temperatures[birth_redshifts > 3],
    }

    # Total number of stars formed
    Num_of_stars_total = len(birth_redshifts)

    for redshift, ax in ax_dict.items():
        data = birth_temperature_by_redshift[redshift]

        H, _ = np.histogram(data, bins=birth_temperature_bins)
        y_points = H / log_birth_temperature_bin_width / Num_of_stars_total

        ax.plot(birth_temperature_centers, y_points, label=name, color=f"C{color}")

        # Add the median stellar birth-pressure line
        ax.axvline(
            np.median(data),
            color=f"C{color}",
            linestyle="dashed",
            zorder=-10,
            alpha=0.5,
        )

axes[0].legend(loc="upper right", markerfirst=False)
axes[2].set_xlabel("Stellar Birth Temperature $T_B$ [K]")
axes[1].set_ylabel("$N_{\\rm bin}$ / d$\\log(T_B)$ / $N_{\\rm total}$")

fig.savefig(f"{arguments.output_directory}/birth_temperature_distribution.png")
