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
from velociraptor.observations import load_observations

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

path_to_obs_data = f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"

observation_lines = []
observation_labels = []

Aird2015 = load_observations(
    [f"{path_to_obs_data}/data/BlackHoleAccretionHistory/Aird2015.hdf5"]
)[0]
observation_lines.append(
    ax.plot(
        Aird2015.x.value,
        Aird2015.y.value,
        color='brown',
        zorder=-10000,
        alpha=0.6,
        linewidth=1,
    )[0]
)
observation_labels.append(Aird2015.citation)

Annana2020_low = load_observations(
    [f"{path_to_obs_data}/data/BlackHoleAccretionHistory/Annana2020_low.hdf5"]
)[0]
observation_lines.append(
    ax.plot(
        Annana2020_low.x.value,
        Annana2020_low.y.value,
        label=Annana2020_low.citation,
        zorder=-10000,
        color='orange',
        alpha=0.6,
        linewidth=1,
    )[0]
)
observation_labels.append(Annana2020_low.citation)

Annana2020_high = load_observations(
    [f"{path_to_obs_data}/data/BlackHoleAccretionHistory/Annana2020_high.hdf5"]
)[0]
observation_lines.append(
    ax.plot(
        Annana2020_high.x.value,
        Annana2020_high.y.value,
        label=Annana2020_high.citation,
        color='olive',
        zorder=-10000,
        alpha=0.6,
        linewidth=1,
    )[0]
)
observation_labels.append(Annana2020_high.citation)

Pouliasis2024_low = load_observations(
    [f"{path_to_obs_data}/data/BlackHoleAccretionHistory/Pouliasis2024_low.hdf5"]
)[0]
observation_lines.append(
    ax.fill_between(
        Pouliasis2024_low.x.value,
        Pouliasis2024_low.y.value - Pouliasis2024_low.y_scatter[0].value,
        Pouliasis2024_low.y.value + Pouliasis2024_low.y_scatter[1].value,
        color='magenta',
        label=Pouliasis2024_low.citation,
        zorder=-10000,
        alpha=0.2,
    )
)
observation_labels.append(Pouliasis2024_low.citation)

Pouliasis2024_high = load_observations(
    [f"{path_to_obs_data}/data/BlackHoleAccretionHistory/Pouliasis2024_high.hdf5"]
)[0]
observation_lines.append(
    ax.fill_between(
        Pouliasis2024_high.x.value,
        Pouliasis2024_high.y.value - Pouliasis2024_high.y_scatter[0].value,
        Pouliasis2024_high.y.value + Pouliasis2024_high.y_scatter[1].value,
        color='teal',
        label=Pouliasis2024_high.citation,
        zorder=-10000,
        alpha=0.2,
    )
)
observation_labels.append(Pouliasis2024_high.citation)

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"BH Accretion Rate Density [M$_\odot$ yr$^{-1}$ Mpc$^{-3}$]")

Delvacchio2014 = load_observations(
    [f"{path_to_obs_data}/data/BlackHoleAccretionHistory/Delvacchio2014.hdf5"]
)[0]
observation_lines.append(
    ax.errorbar(
        Delvacchio2014.x.value,
        Delvacchio2014.y.value,
        xerr=None if Delvacchio2014.x_scatter is None else Delvacchio2014.x_scatter.value,
        yerr=None if Delvacchio2014.y_scatter is None else Delvacchio2014.y_scatter.value,
        label=Delvacchio2014.citation,
        linestyle="none",
        marker="o",
        color="maroon",
        elinewidth=0.75,
        markeredgecolor="none",
        markersize=3,
        zorder=-10,
    )[0]
)
observation_labels.append(Delvacchio2014.citation)

DSilva2023 = load_observations(
    [f"{path_to_obs_data}/data/BlackHoleAccretionHistory/DSilva2023.hdf5"]
)[0]
observation_lines.append(
    ax.errorbar(
        DSilva2023.x.value,
        DSilva2023.y.value,
        xerr=None if DSilva2023.x_scatter is None else DSilva2023.x_scatter.value,
        yerr=None if DSilva2023.y_scatter is None else DSilva2023.y_scatter.value,
        label=DSilva2023.citation,
        linestyle="none",
        marker="o",
        color="darkblue",
        elinewidth=0.75,
        markeredgecolor="none",
        markersize=3,
        zorder=-10,
    )[0]
)
observation_labels.append(DSilva2023.citation)

Yang2023 = load_observations(
    [f"{path_to_obs_data}/data/BlackHoleAccretionHistory/Yang2023.hdf5"]
)[0]
observation_lines.append(
    ax.errorbar(
        Yang2023.x.value,
        Yang2023.y.value,
        xerr=None if Yang2023.x_scatter is None else Yang2023.x_scatter.value,
        yerr=None if Yang2023.y_scatter is None else Yang2023.y_scatter.value,
        label=Yang2023.citation,
        linestyle="none",
        marker="o",
        color="goldenrod",
        elinewidth=0.75,
        markeredgecolor="none",
        markersize=3,
        zorder=-10,
    )
)
observation_labels.append(Yang2023.citation)

Shen2020_low = load_observations(
    [f"{path_to_obs_data}/data/BlackHoleAccretionHistory/Shen2020_low.hdf5"]
)[0]
observation_lines.append(
    ax.errorbar(
        Shen2020_low.x.value,
        Shen2020_low.y.value,
        xerr=None if Shen2020_low.x_scatter is None else Shen2020_low.x_scatter.value,
        yerr=None if Shen2020_low.y_scatter is None else Shen2020_low.y_scatter.value,
            label=Shen2020_low.citation,
            linestyle="none",
            marker="o",
            color="lime",
            elinewidth=0.5,
            markeredgecolor="none",
            markersize=2,
            zorder=-10,
    )
)
observation_labels.append(Shen2020_low.citation)

Shen2020_high = load_observations(
    [f"{path_to_obs_data}/data/BlackHoleAccretionHistory/Shen2020_high.hdf5"]
)[0]
observation_lines.append(
    ax.errorbar(
        Shen2020_high.x.value,
        Shen2020_high.y.value,
        xerr=None if Shen2020_high.x_scatter is None else Shen2020_high.x_scatter.value,
        yerr=None if Shen2020_high.y_scatter is None else Shen2020_high.y_scatter.value,
            label=Shen2020_high.citation,
            linestyle="none",
            marker="o",
            color="coral",
            elinewidth=0.5,
            markeredgecolor="none",
            markersize=2,
            zorder=-10,
    )
)
observation_labels.append(Shen2020_high.citation)

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
ax.set_ylim(1e-7, 6e-4)

#observation_legend = ax.legend(markerfirst=True, loc="lower left")
observation_legend = ax.legend(
    observation_lines, observation_labels, markerfirst=True, loc="lower center", fontsize=4, ncol=1
)

ax.add_artist(observation_legend)

simulation_legend = ax.legend(
    simulation_lines, simulation_labels, markerfirst=False, loc="upper right"
)

ax.add_artist(simulation_legend)

fig.savefig(f"{output_path}/bh_accretion_evolution.png")
