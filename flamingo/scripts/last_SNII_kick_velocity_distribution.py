"""
Plots the SNII kick velocity distribution (at last SNII kinetic-feedback events).
"""

import matplotlib.pyplot as plt
import numpy as np
import unyt

from swiftsimio import load
from swiftpipeline.argumentparser import ScriptArgumentParser

arguments = ScriptArgumentParser(
    description="Creates a plot showing the distribution of SNII kick velocities "
    "recorded when the gas was last kicked by SNII, split by redshift"
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

SNII_v_kick_bins = unyt.unyt_array(np.logspace(-1, 4, number_of_bins), units="km/s")
log_SNII_v_kick_bin_width = np.log10(SNII_v_kick_bins[1].value) - np.log10(
    SNII_v_kick_bins[0].value
)
SNII_v_kick_centres = 0.5 * (SNII_v_kick_bins[1:] + SNII_v_kick_bins[:-1])

# Begin plotting
fig, axes = plt.subplots(1, 1, sharex=True, sharey=True)

ax_dict = {"$z > 0$": axes}

for label, ax in ax_dict.items():
    ax.loglog()
    ax.text(0.025, 1.0 - 0.025 * 3, label, transform=ax.transAxes, ha="left", va="top")

for color, (snapshot, name) in enumerate(zip(data, names)):

    try:
        stars_SNII_v_kick_last = (snapshot.stars.last_sniikinetic_feedbackvkick).to(
            SNII_v_kick_bins.units
        )

        gas_SNII_v_kick_last = (snapshot.gas.last_sniikinetic_feedbackvkick).to(
            SNII_v_kick_bins.units
        )
    except AttributeError:
        print("No tracer data")
        continue

    stars_SNII_v_kick_last = stars_SNII_v_kick_last
    gas_SNII_v_kick_last = gas_SNII_v_kick_last

    data = np.concatenate(
        (
            stars_SNII_v_kick_last[stars_SNII_v_kick_last > -1],
            gas_SNII_v_kick_last[gas_SNII_v_kick_last > -1],
        )
    )

    Num_of_kicked_parts_total = len(data)

    H, _ = np.histogram(data, bins=SNII_v_kick_bins)
    y_points = H / log_SNII_v_kick_bin_width / Num_of_kicked_parts_total

    ax.plot(SNII_v_kick_centres, y_points, label=name, color=f"C{color}")
    ax.axvline(
        np.median(data), color=f"C{color}", linestyle="dashed", zorder=-10, alpha=0.5
    )

axes.legend(loc="upper right", markerfirst=False)
axes.set_xlabel("Kick velocity at last SNII $v_{\\rm kick}$ [km s$^{-1}$]")
axes.set_ylabel("$N_{\\rm bin}$ / d$\\log v_{\\rm kick}$ / $N_{\\rm total}$")

fig.savefig(f"{arguments.output_directory}/SNII_last_kick_velocity_distribution.png")
