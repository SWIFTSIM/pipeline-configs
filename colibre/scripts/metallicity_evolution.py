"""
Plots the gas and stellar metallicity density evolution.
"""
import matplotlib

import unyt

import matplotlib.pyplot as plt
import numpy as np
import sys


from swiftsimio import load, load_statistics

from swiftpipeline.argumentparser import ScriptArgumentParser

arguments = ScriptArgumentParser(
    description="Creates a metallicity density evolution plot for gas, stars and black holes."
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

for color, (snapshot_filename, stats_filename, name) in enumerate(
    zip(snapshot_filenames, stats_filenames, names)
):
    data = load_statistics(stats_filename)

    snapshot = load(snapshot_filename)
    boxsize = snapshot.metadata.boxsize.to("Mpc")
    box_volume = boxsize[0] * boxsize[1] * boxsize[2]

    # a, Redshift, SFR
    scale_factor = data.a
    redshift = data.z
    gas_Z_mass = data.gas_z_mass.to("Msun")
    star_Z_mass = data.star_z_mass.to("Msun")
    gas_Z_mass_density = gas_Z_mass / box_volume
    star_Z_mass_density = star_Z_mass / box_volume

    # High z-order as we always want these to be on top of the observations
    simulation_lines.append(
        ax.plot(
            scale_factor,
            gas_Z_mass_density,
            linestyle="solid",
            color=f"C{color}",
            zorder=10000,
        )[0]
    )

    # Stellar metallicity not used as a line.
    ax.plot(
        scale_factor,
        star_Z_mass_density,
        linestyle="dashed",
        color=f"C{color}",
        zorder=10000,
    )

    simulation_labels.append(name)

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"Metal Mass $\rho_{\rm Z}$ [M$_\odot$ Mpc$^{-3}$]")

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
ax.set_ylim(3e4, 4e7)

from matplotlib.lines import Line2D

custom_lines = [
    Line2D([0], [0], color="black", linestyle="solid"),
    Line2D([0], [0], color="black", linestyle="dashed"),
]
custom_legend = ax.legend(
    custom_lines, ["Gas", "Stars"], markerfirst=True, loc="lower left"
)

ax.add_artist(custom_legend)

simulation_legend = ax.legend(
    simulation_lines, simulation_labels, markerfirst=False, loc="upper right"
)

ax.add_artist(simulation_legend)

fig.savefig(f"{output_path}/metallicity_evolution.png")
