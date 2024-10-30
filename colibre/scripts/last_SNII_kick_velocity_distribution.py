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
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
axes = axes.flat

z = data[0].metadata.z

if z < 5:
    ax_dict = {"$z < 1$": axes[0], "$1 < z < 3$": axes[1], "$z > 3$": axes[2]}
else:
    ax_dict = {"$z < 7$": axes[0], "$7 < z < 10$": axes[1], "$z > 10$": axes[2]}

for label, ax in ax_dict.items():
    ax.loglog()
    ax.text(0.025, 1.0 - 0.025 * 3, label, transform=ax.transAxes, ha="left", va="top")

for color, (snapshot, name) in enumerate(zip(data, names)):

    stars_SNII_v_kick_last = (snapshot.stars.last_sniikinetic_feedbackvkick).to(
        SNII_v_kick_bins.units
    )

    stars_SNII_redshifts = (
        1 / snapshot.stars.last_sniikinetic_feedback_scale_factors.value - 1
    )

    gas_SNII_v_kick_last = (snapshot.gas.last_sniikinetic_feedbackvkick).to(
        SNII_v_kick_bins.units
    )

    gas_SNII_redshifts = (
        1 / snapshot.gas.last_sniikinetic_feedback_scale_factors.value - 1
    )

    # Limit only to those gas / stellar particles that were in fact kicked by SNII
    stars_SNII_kicked = stars_SNII_v_kick_last > 0.0
    gas_SNII_kicked = gas_SNII_v_kick_last > 0.0

    # Select only those parts that were kicked by SNII in the past
    stars_SNII_v_kick_last = stars_SNII_v_kick_last[stars_SNII_kicked]
    stars_SNII_redshifts = stars_SNII_redshifts[stars_SNII_kicked]
    gas_SNII_v_kick_last = gas_SNII_v_kick_last[gas_SNII_kicked]
    gas_SNII_redshifts = gas_SNII_redshifts[gas_SNII_kicked]

    # Segment SNII kick velocities into redshift bins
    if z < 5:
        stars_SNII_v_kick_by_redshift = {
            "$z < 1$": stars_SNII_v_kick[birth_redshifts < 1],
            "$1 < z < 3$": stars_SNII_v_kick[
                np.logical_and(birth_redshifts > 1, birth_redshifts < 3)
            ],
            "$z > 3$": stars_SNII_v_kick[birth_redshifts > 3],
        }

        gas_SNII_v_kick_by_redshift = {
            "$z < 1$": gas_SNII_v_kick[birth_redshifts < 1],
            "$1 < z < 3$": gas_SNII_v_kick[
                np.logical_and(birth_redshifts > 1, birth_redshifts < 3)
            ],
            "$z > 3$": gas_SNII_v_kick[birth_redshifts > 3],
        }
    else:
        stars_SNII_v_kick_by_redshift = {
            "$z < 7$": stars_SNII_v_kick[birth_redshifts < 7],
            "$7 < z < 10$": stars_SNII_v_kick[
                np.logical_and(birth_redshifts > 7, birth_redshifts < 10)
            ],
            "$z > 10$": stars_SNII_v_kick[birth_redshifts > 10],
        }

        gas_SNII_v_kick_by_redshift = {
            "$z < 7$": gas_SNII_v_kick[birth_redshifts < 7],
            "$7 < z < 10$": gas_SNII_v_kick[
                np.logical_and(birth_redshifts > 7, birth_redshifts < 10)
            ],
            "$z > 10$": gas_SNII_v_kick[birth_redshifts > 10],
        }

    # Fetch target kick velocity
    SNII_target_kick_velocity_km_p_s = float(
        snapshot.metadata.parameters["COLIBREFeedback:SNII_delta_v_km_p_s"].decode(
            "utf-8"
        )
    )  # in km/s

    # Total number of objects received SNII kinetic energy
    Num_of_kicked_parts_total = len(gas_SNII_redshifts) + len(stars_SNII_redshifts)

    for redshift, ax in ax_dict.items():
        data = np.concatenate(
            [
                stars_SNII_v_kick_by_redshift[redshift],
                gas_SNII_v_kick_by_redshift[redshift],
            ]
        )

        H, _ = np.histogram(data, bins=SNII_v_kick_bins)
        y_points = H / log_SNII_v_kick_bin_width / Num_of_kicked_parts_total

        ax.plot(SNII_v_kick_centres, y_points, label=name, color=f"C{color}")
        ax.axvline(
            np.median(data),
            color=f"C{color}",
            linestyle="dashed",
            zorder=-10,
            alpha=0.5,
        )

        # Add the line indicating the target kick velocity
        ax.axvline(
            SNII_target_kick_velocity_km_p_s,
            color=f"C{color}",
            linestyle="dotted",
            zorder=-10,
            alpha=0.5,
        )

axes[0].legend(loc="upper right", markerfirst=False)
axes[2].set_xlabel("Kick velocity at last SNII $v_{\\rm kick}$ [km s$^{-1}$]")
axes[1].set_ylabel("$N_{\\rm bin}$ / d$\\log v_{\\rm kick}$ / $N_{\\rm total}$")

fig.savefig(f"{arguments.output_directory}/SNII_last_kick_velocity_distribution.png")
