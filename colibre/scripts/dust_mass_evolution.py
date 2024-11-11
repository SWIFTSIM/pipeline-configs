"""
Plots the dust mass density evolution.
"""

import matplotlib.pyplot as plt
import numpy as np
import glob


from swiftsimio import load, load_statistics

from swiftpipeline.argumentparser import ScriptArgumentParser

from velociraptor.observations import load_observations

arguments = ScriptArgumentParser(
    description="Creates a dust mass density evolution plot, with added observational data."
)

snapshot_filenames = [
    f"{directory}/{snapshot}"
    for directory, snapshot in zip(arguments.directory_list, arguments.snapshot_list)
]

stats_filenames = [
    f"{directory}/statistics.txt" for directory in arguments.directory_list
]

names = arguments.name_list
output_path = arguments.output_directory

plt.style.use(arguments.stylesheet_location)

simulation_lines = []
simulation_labels = []

fig, ax = plt.subplots()

ax.loglog()

for snapshot_filename, stats_filename, name in zip(
    snapshot_filenames, stats_filenames, names
):
    data = load_statistics(stats_filename)

    snapshot = load(snapshot_filename)
    boxsize = snapshot.metadata.boxsize.to("Mpc")
    box_volume = boxsize[0] * boxsize[1] * boxsize[2]

    # a, Redshift, SFR
    scale_factor = data.a
    redshift = data.z
    dust_mass = data.dust_mass.to("Msun")
    dust_mass_density = dust_mass / box_volume

    # High z-order as we always want these to be on top of the observations
    simulation_lines.append(ax.plot(scale_factor, dust_mass_density, zorder=10000)[0])
    simulation_labels.append(name)

# Observational data plotting
path_to_obs_data = f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"
observational_data = load_observations(
    sorted(glob.glob(f"{path_to_obs_data}/data/DustMassDensity/*.hdf5"))
)

for obs_data in observational_data:
    obs_data.plot_on_axes(ax)

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"Dust Mass Density $\rho_*$ [M$_\odot$ Mpc$^{-3}$]")

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
ax.tick_params(axis="x", which="minor", bottom=False)

ax.set_xlim(1.02, 0.07)
ax.set_ylim(1e3, 1.3e6)

observation_legend = ax.legend(markerfirst=True, loc="lower left")

ax.add_artist(observation_legend)

simulation_legend = ax.legend(
    simulation_lines, simulation_labels, markerfirst=False, loc="upper right"
)

ax.add_artist(simulation_legend)

fig.savefig(f"{output_path}/dust_mass_evolution.png")
