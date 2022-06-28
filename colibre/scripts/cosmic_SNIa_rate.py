"""
Plots the cosmic SNIa rate history.
"""
import unyt

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load

from load_sfh_data import read_obs_data

from velociraptor.observations import load_observations

from astropy.cosmology import z_at_value
from astropy.units import Gyr

SNIa_rate_output_units = 1.0 / (unyt.yr * unyt.Mpc ** 3)

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

ax.loglog()

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
    simulation_lines.append(ax.plot(scale_factor, SNIa_rate.value, zorder=10000)[0])
    simulation_labels.append(name)


observation_lines = []
observation_labels = []

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
    observation_lines, observation_labels, markerfirst=True, loc=3, fontsize=4, ncol=2
)

simulation_legend = ax.legend(
    simulation_lines, simulation_labels, markerfirst=False, loc="upper right"
)

ax.add_artist(observation_legend)

# Create second X-axis (to plot cosmic time alongside redshift)
ax2 = ax.twiny()
ax2.set_xscale("log")

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

ax.set_ylim(1.0e-6, 1.0e-4)
ax.set_xlim(1.02, 0.07)
ax2.set_xlim(1.02, 0.07)

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"SNIa rate [yr$^{-1}$ Mpc$^{-3}$]")
ax2.set_xlabel("Cosmic time [Gyr]")

fig.savefig(f"{output_path}/SNIa_rate_history.png")
