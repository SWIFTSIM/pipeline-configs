"""
Plots the molecular gas mass density evolution.
"""
import matplotlib

import unyt

import matplotlib.pyplot as plt
import numpy as np
import sys
import glob

from unyt import unyt_quantity

from swiftsimio import load, load_statistics

from swiftpipeline.argumentparser import ScriptArgumentParser

from velociraptor.observations import load_observation

arguments = ScriptArgumentParser(
    description="Creates a molecular gas mass density evolution plot, with added observational data."
)

snapshot_filenames = [
    f"{directory}/{snapshot}"
    for directory, snapshot in zip(arguments.directory_list, arguments.snapshot_list)
]

stats_filenames = [
    f"{directory}/statistics.txt" for directory in arguments.directory_list
]

sfh_filenames = [
    f"{directory}/SFR.txt" for directory in arguments.directory_list
]


names = arguments.name_list
output_path = arguments.output_directory

plt.style.use(arguments.stylesheet_location)

simulation_lines = []
simulation_labels = []

fig, ax = plt.subplots()

ax.loglog()

for snapshot_filename, stats_filename, sfh_filename, name in zip(
        snapshot_filenames, stats_filenames, sfh_filenames, names
):
    data = load_statistics(stats_filename)
    Mform = np.cumsum(np.genfromtxt(sfh_filename)[:,4])[data.step]*1e10 * 0.6
    snapshot = load(snapshot_filename)
    boxsize = snapshot.metadata.boxsize.to("Mpc")
    box_volume = boxsize[0] * boxsize[1] * boxsize[2]
    cosmo = snapshot.metadata.cosmology

    # a, Redshift, SFR
    #print(dir()
    scale_factor = data.a
    redshift = data.z
    H2_mass = data.gas_h2_mass.to("Msun")

    # Convert to abundance
    H2tostellar = H2_mass/Mform

    # High z-order as we always want these to be on top of the observations
    simulation_lines.append(ax.plot(scale_factor, H2tostellar, zorder=10000)[0])
    simulation_labels.append(name)

# Observational data plotting
T18_data = np.genfromtxt(
    f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"
    "/data/CosmicH2toStellarFraction/raw/Tacconi18_h2toMstar_evo.txt")
T18_expansion_factor = pow(10., -T18_data[:,0])
T18_H2tostellar_frac = pow(10., T18_data[:,1])

simulation_lines.append(ax.errorbar(T18_expansion_factor,
                                    T18_H2tostellar_frac, 0, marker='o',
                                    label="Tacconi et. al (2021)",ls='none',
                                   color="C3"))

# observational_data = glob.glob(
#    f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}/data/StellarMassDensity/*.hdf5"
# )

# for index, observation in enumerate(observational_data):
#    obs = load_observation(observation)
#    obs.plot_on_axes(ax)

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"$\Omega_{H2}/\Omega_{\star}$ [-]")

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
#ax.set_ylim(3e-6, 8e-2)

observation_legend = ax.legend(markerfirst=True, loc="lower left")

ax.add_artist(observation_legend)

simulation_legend = ax.legend(
    simulation_lines, simulation_labels, markerfirst=False, loc="center right"
)

ax.add_artist(simulation_legend)

fig.savefig(f"{output_path}/H2tostellar_mass_evolution.png")
