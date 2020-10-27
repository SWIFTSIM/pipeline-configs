"""
Plots the BH accretion rate density evolution
"""
import matplotlib

import unyt

import matplotlib.pyplot as plt
import numpy as np
import sys
import glob


from swiftsimio import load, load_statistics

from swiftpipeline.argumentparser import ScriptArgumentParser

from velociraptor.observations import load_observation

arguments = ScriptArgumentParser(
    description="Creates a BH accretion rate density evolution plot, with added observational data."
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

    # a, Redshift, BHARD
    scale_factor = data.a
    redshift = data.z
    bh_instant_accretion_rate = data.bh_acc_rate.to("Msun / yr")
    bh_instant_accretion_rate_density = bh_instant_accretion_rate / box_volume

    # Compute BHAR from BH accreted mass and time
    time = data.time.to("yr")
    bh_accreted_mass = data.bh_acc_mass.to("Msun")
    bh_accretion_rate = (bh_accreted_mass[1:] - bh_accreted_mass[:-1]) / (
        time[1:] - time[:-1]
    )
    bh_accretion_rate_density = bh_accretion_rate / box_volume

    # High z-order as we always want these to be on top of the observations
    simulation_lines.append(
        ax.plot(
            0.5 * (scale_factor[1:] + scale_factor[:-1]),
            bh_accretion_rate_density,
            zorder=10000,
        )[0]
    )
    simulation_labels.append(name)

# Observational data plotting

observational_data = glob.glob(
    f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}/data/BlackHoleAccretionHistory/*.hdf5"
)

for index, observation in enumerate(observational_data):
    obs = load_observation(observation)
    obs.plot_on_axes(ax)

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"BH Accretion Rate Density [M$_\odot$ yr$^{-1}$ Mpc$^{-3}$]")

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
ax.set_ylim(6e-7, 6e-4)

observation_legend = ax.legend(markerfirst=True, loc="lower left")

ax.add_artist(observation_legend)

simulation_legend = ax.legend(
    simulation_lines, simulation_labels, markerfirst=False, loc="upper right"
)

ax.add_artist(simulation_legend)

fig.savefig(f"{output_path}/bh_accretion_evolution.png")
