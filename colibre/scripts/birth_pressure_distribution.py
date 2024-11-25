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
    description="Creates a stellar birth pressure distribution plot, split by redshift"
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

birth_pressure_bins = unyt.unyt_array(
    np.logspace(1.0, 8.0, number_of_bins), units="K/cm**3"
)
log_birth_pressure_bin_width = np.log10(birth_pressure_bins[1].value) - np.log10(
    birth_pressure_bins[0].value
)
birth_pressure_centers = 0.5 * (birth_pressure_bins[1:] + birth_pressure_bins[:-1])


# Begin plotting
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
axes = axes.flat

z = data[0].metadata.z

if z < 4.9:
    ax_dict = {"$z < 1$": axes[0], "$1 < z < 3$": axes[1], "$z > 3$": axes[2]}
else:
    ax_dict = {"$z < 7$": axes[0], "$7 < z < 10$": axes[1], "$z > 10$": axes[2]}

for label, ax in ax_dict.items():
    ax.loglog()
    ax.text(0.025, 1.0 - 0.025 * 3, label, transform=ax.transAxes, ha="left", va="top")

for color, (snapshot, name) in enumerate(zip(data, names)):

    birth_densities = snapshot.stars.birth_densities.to("g/cm**3") / mh.to("g")
    birth_temperatures = snapshot.stars.birth_temperatures.to("K")
    birth_pressures = (birth_densities * birth_temperatures).to("K/cm**3")
    birth_redshifts = 1 / snapshot.stars.birth_scale_factors.value - 1

    # Segment birth pressures into redshift bins
    if z < 4.9:
        birth_pressure_by_redshift = {
            "$z < 1$": birth_pressures[birth_redshifts < 1],
            "$1 < z < 3$": birth_pressures[
                np.logical_and(birth_redshifts > 1, birth_redshifts < 3)
            ],
            "$z > 3$": birth_pressures[birth_redshifts > 3],
        }
    else:
        birth_pressure_by_redshift = {
            "$z < 7$": birth_pressures[birth_redshifts < 7],
            "$7 < z < 10$": birth_pressures[
                np.logical_and(birth_redshifts > 7, birth_redshifts < 10)
            ],
            "$z > 10$": birth_pressures[birth_redshifts > 10],
        }

    # Total number of stars formed
    Num_of_stars_total = len(birth_redshifts)

    for redshift, ax in ax_dict.items():
        data = birth_pressure_by_redshift[redshift]
        if data.shape[0] == 0:
            continue

        H, _ = np.histogram(data, bins=birth_pressure_bins)
        y_points = H / log_birth_pressure_bin_width / Num_of_stars_total

        ax.plot(birth_pressure_centers, y_points, label=name, color=f"C{color}")

        # Add the median stellar birth-pressure line
        ax.axvline(
            np.median(data),
            color=f"C{color}",
            linestyle="dashed",
            zorder=-10,
            alpha=0.5,
        )

axes[0].legend(loc="upper right", markerfirst=False)
axes[2].set_xlabel("Stellar Birth Pressure $P_B/k$ [K cm$^{-3}$]")
axes[1].set_ylabel("$N_{\\rm bin}$ / d$\\log(P_B/k)$ / $N_{\\rm total}$")

fig.savefig(f"{arguments.output_directory}/birth_pressure_distribution.png")
