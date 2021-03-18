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
    cosmo = snapshot.metadata.cosmology
    Omega_b = cosmo.Ob0
    rho_crit0 = unyt_quantity.from_astropy(cosmo.critical_density0)
    rho_crit0 = rho_crit0.to("Msun / Mpc**3")

    # a, Redshift, SFR
    scale_factor = data.a
    redshift = data.z
    H2_mass = data.gas_h2_mass.to("Msun")
    H2_mass_density = H2_mass / box_volume

    # Convert to abundance
    Omega_H2 = H2_mass_density / rho_crit0
    # High z-order as we always want these to be on top of the observations
    simulation_lines.append(ax.plot(scale_factor, Omega_H2, zorder=10000)[0])
    simulation_labels.append(name)

    ax.axhline(Omega_b, c="0.4", ls="--", lw=0.5)
# Observational data plotting
S18_data = (10.**np.genfromtxt(
    f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"
    "/data/CosmicH2Abundance/raw/Saintonge18_CH2D.txt"))/rho_crit0.value
simulation_lines.append(ax.errorbar(1, S18_data[0],
                                    np.row_stack([S18_data[0]-S18_data[1], S18_data[2]-S18_data[0]]),
                                    marker='o', label="Saintonge et. al (2018)"))

S17_data = np.genfromtxt(
    f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"
    "/data/CosmicH2Abundance/raw/Scoville17_CH2D.txt")
S17_expansion_factor = 1./(1+S17_data[:,0])
S17_Omega_H2 = pow(10.,S17_data[:,1])/rho_crit0.value
simulation_lines.append(ax.errorbar(S17_expansion_factor, S17_Omega_H2, 0, ls='none',
                                    marker='o', label="Scoville et. al (2017)"))

F21_data = np.genfromtxt(
    f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"
    "/data/CosmicH2Abundance/raw/Fletcher2021_H2.txt")/snapshot.metadata.cosmology.h
simulation_lines.append(ax.errorbar(1, F21_data[0],
                                    np.row_stack([F21_data[0]-F21_data[1], F21_data[2]-F21_data[0]]),
                                    marker='o', label="Fletcher et. al (2021)"))

P20_data = np.genfromtxt(
    f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"
    "/data/CosmicH2Abundance/raw/Peroux2020_OmegaH2.txt")
simulation_lines.append(ax.errorbar(1./(P20_data[:,0]+1.), 10**P20_data[:,1],
                                    np.row_stack([10**P20_data[:,1] - 10**P20_data[:,2],
                                                  10**P20_data[:,3] - 10**P20_data[:,1]]),
                                    zorder=0, fmt='o', 
                                    marker='o', label="Peroux & Howk (2020)"))

G21_data = np.genfromtxt(
    f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"
    "/data/CosmicH2Abundance/raw/Garrat21_OmegaH2.txt")
simulation_lines.append(ax.errorbar(pow(G21_data[:,0]+1,-1), G21_data[:,1]*1e-7,
                                    np.row_stack([G21_data[:,2], G21_data[:,2]])*1e-7,
                                    zorder=0, fmt='o', 
                                    marker='o', label="Garrat et al. (2021)"))

zgrid = np.linspace(0,4, 50)

W20_data = np.genfromtxt(
    f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"
    "/data/CosmicH2Abundance/raw/Walter2020_rhoH2.txt", usecols=[1,2,3,4])
W20eq1 = lambda z,a,b,c,d: ((a*(1+z)**b)/ (1+((1+z)/c)**d)) / rho_crit0
W20_rhoH2 = W20eq1(zgrid, *W20_data[0])
W20_rhoH2_lo = W20eq1(zgrid, *W20_data[1])
W20_rhoH2_hi = W20eq1(zgrid, *W20_data[2])

simulation_lines.append(ax.fill_between(pow(1+zgrid,-1),
                                        W20_rhoH2_lo,
                                        W20_rhoH2_hi,
                                        alpha=0.2,
                                        color='C6',
                                        label="Walter et al. (2020) Fit"))
simulation_lines.append(ax.plot(pow(1+zgrid,-1),
                                W20_rhoH2,
                                color='C6'))


# observational_data = glob.glob(
#    f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}/data/StellarMassDensity/*.hdf5"
# )

# for index, observation in enumerate(observational_data):
#    obs = load_observation(observation)
#    obs.plot_on_axes(ax)

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"Molecular Gas Cosmic Abundance $\Omega_{\rm H2}$ [-]")

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
ax.set_ylim(3e-7, 4e-3)

observation_legend = ax.legend(markerfirst=True, loc="lower left")

ax.add_artist(observation_legend)

simulation_legend = ax.legend(
    simulation_lines, simulation_labels, markerfirst=False, loc="center right"
)

ax.add_artist(simulation_legend)

fig.savefig(f"{output_path}/H2_mass_evolution.png")
