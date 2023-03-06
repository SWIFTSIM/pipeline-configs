"""
Plots the cosmic SNIa rate history.
"""
import unyt

import matplotlib.pyplot as plt
import numpy as np
import glob

from swiftsimio import load

from velociraptor.observations import load_observations

from astropy.cosmology import z_at_value
from astropy.units import Gyr

from swiftpipeline.argumentparser import ScriptArgumentParser

arguments = ScriptArgumentParser(
    description="Creates a SNIa rate history plot, with added observational data."
)

snapshot_filenames = [
    f"{directory}/{snapshot}"
    for directory, snapshot in zip(arguments.directory_list, arguments.snapshot_list)
]

SNIa_filenames = [f"{directory}/SNIa.txt" for directory in arguments.directory_list]

names = arguments.name_list
output_path = arguments.output_directory

plt.style.use(arguments.stylesheet_location)

simulation_lines = []
simulation_labels = []

fig, ax = plt.subplots()

ax.semilogx()

log_multiplicative_factor = 4
multiplicative_factor = 10 ** log_multiplicative_factor
SNIa_rate_output_units = 1.0 / (unyt.yr * unyt.Mpc ** 3)

for idx, (snapshot_filename, SNIa_filename, name) in enumerate(
    zip(snapshot_filenames, SNIa_filenames, names)
):
    data = np.loadtxt(
        SNIa_filename,
        usecols=(4, 6, 11),
        dtype=[("a", np.float32), ("z", np.float32), ("SNIa rate", np.float32)],
    )

    snapshot = load(snapshot_filename)

    # Read cosmology from the first run in the list
    if idx == 0:
        cosmology = snapshot.metadata.cosmology

    units = snapshot.units
    SNIa_rate_units = 1.0 / (units.time * units.length ** 3)

    # a, Redshift, SFR
    scale_factor = data["a"]
    SNIa_rate = (data["SNIa rate"] * SNIa_rate_units).to(SNIa_rate_output_units)

    # High z-order as we always want these to be on top of the observations
    simulation_lines.append(
        ax.plot(scale_factor, SNIa_rate.value * multiplicative_factor, zorder=10000)[0]
    )
    simulation_labels.append(name)


observation_lines = []
observation_labels = []

path_to_obs_data = f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"
observational_data = load_observations(
    sorted(glob.glob(f"{path_to_obs_data}/data/CosmicSNIaRate/*.hdf5"))
)

for obs_data in observational_data:
    observation_lines.append(
        ax.errorbar(
            obs_data.x.value,
            obs_data.y.value * multiplicative_factor,
            yerr=obs_data.y_scatter.value * multiplicative_factor,
            label=obs_data.citation,
            linestyle="none",
            marker="o",
            elinewidth=0.5,
            markeredgecolor="none",
            markersize=2,
            zorder=-10,
            capsize=1.0,
        )
    )
    observation_labels.append(f"{obs_data.citation}")

redshift_ticks = np.array([0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0])
redshift_labels = [
    "$0$",
    "$0.2$",
    "$0.5$",
    "$1$",
    "$2$",
    "$3$",
    "$5$",
    "$10$",
    "$20$",
    "$50$",
    "$100$",
]
a_ticks = 1.0 / (redshift_ticks + 1.0)

ax.set_xticks(a_ticks)
ax.set_xticklabels(redshift_labels)

observation_legend = ax.legend(
    observation_lines, observation_labels, markerfirst=True, loc="lower right", fontsize="xx-small", ncol=2
)

simulation_legend = ax.legend(
    simulation_lines, simulation_labels, markerfirst=False, loc="upper right"
)

ax.add_artist(observation_legend)

# Create second X-axis (to plot cosmic time alongside redshift)
ax2 = ax.twiny()
ax2.set_xscale("log")
ax.set_yscale("log")

# Cosmic-time ticks (in Gyr) along the second X-axis
t_ticks = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, cosmology.age(1.0e-5).value])

# To place the new ticks onto the X-axis we need to know the corresponding scale factors
a_ticks_2axis = [
    1.0 / (1.0 + z_at_value(cosmology.age, t_tick * Gyr)) for t_tick in t_ticks
]

# Attach the ticks to the second X-axis
ax2.set_xticks(a_ticks_2axis)

# Format the ticks' labels
ax2.set_xticklabels(["$%2.1f$" % t_tick for t_tick in t_ticks])

# Final adjustments
ax.tick_params(axis="x", which="minor", bottom=False)
ax2.tick_params(axis="x", which="minor", top=False)

ax.set_ylim(3e-2, 2.0)
ax.set_xlim(1.02, 0.07)
ax2.set_xlim(1.02, 0.07)

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(
    f"SNIa rate [$10^{{-{log_multiplicative_factor}}}$ yr$^{{-1}}$ cMpc$^{{-3}}$]"
)
ax2.set_xlabel("Cosmic time [Gyr]")

fig.savefig(f"{output_path}/log_SNIa_rate_history.png")
