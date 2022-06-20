"""
Plots the atomic gas mass density evolution.
"""
import matplotlib

import unyt

import matplotlib.pyplot as plt
import numpy as np

from unyt import unyt_quantity

from swiftsimio import load, load_statistics

from swiftpipeline.argumentparser import ScriptArgumentParser

from velociraptor.observations import load_observations

arguments = ScriptArgumentParser(
    description="Creates a atomic gas mass density evolution plot, with added observational data."
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
    rho_crit0 = unyt_quantity.from_astropy(cosmo.critical_density0)
    rho_crit0 = rho_crit0.to("Msun / Mpc**3")

    # a, Redshift, SFR
    scale_factor = data.a
    redshift = data.z
    HI_mass = data.gas_hi_mass.to("Msun")
    HI_mass_density = HI_mass / box_volume

    # High z-order as we always want these to be on top of the observations
    simulation_lines.append(ax.plot(scale_factor, HI_mass_density, zorder=10000)[0])
    simulation_labels.append(name)

# Observational data plotting

observation_lines = []
observation_labels = []

zgrid = np.linspace(0, 4, 50)
agrid = pow(1 + zgrid, -1)

P20_data = np.genfromtxt(
    f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"
    "/data/CosmicHIAbundance/raw/Peroux2020_OmegaHI.txt",
    usecols=[1, 2],
)
P20eq13 = lambda z, a, b: a * rho_crit0 * (1.0 + z) ** b
P20_rhoHI = P20eq13(zgrid, *P20_data[0])
P20_rhoHI_lo = P20eq13(zgrid, *P20_data[1])
P20_rhoHI_hi = P20eq13(zgrid, *P20_data[2])

# Cosmology correction (Schaye, 2001, ApJ 559, 507)
# This correction derives from the fact that the measurements have been homogenised
# assuming a cosmology with h=0.7, Omega_M = 0.3 and Omega_Lambda = 0.7
# This affects the conversion from redshift distances into actual co-moving distances:
# Delta(X) ~ H_0/H(z) = [Omega_Lambda + Omega_M * (1+z)^3]^(-1/2)
P20_rhoHI *= cosmo.h ** -1 * np.sqrt(cosmo.Om0 * (1.0 + zgrid) ** 3 + cosmo.Ode0)
P20_rhoHI /= 0.7 ** -1 * np.sqrt(0.3 * (1.0 + zgrid) ** 3 + 0.7)
P20_rhoHI_lo *= cosmo.h ** -1 * np.sqrt(cosmo.Om0 * (1.0 + zgrid) ** 3 + cosmo.Ode0)
P20_rhoHI_lo /= 0.7 ** -1 * np.sqrt(0.3 * (1.0 + zgrid) ** 3 + 0.7)
P20_rhoHI_hi *= cosmo.h ** -1 * np.sqrt(cosmo.Om0 * (1.0 + zgrid) ** 3 + cosmo.Ode0)
P20_rhoHI_hi /= 0.7 ** -1 * np.sqrt(0.3 * (1.0 + zgrid) ** 3 + 0.7)

# convert from neutral gas density (H+He) to HI (only H) mass density
P20_rhoHI *= 0.76
P20_rhoHI_lo *= 0.76
P20_rhoHI_hi *= 0.76

ax.fill_between(
    agrid,
    P20_rhoHI_lo,
    P20_rhoHI_hi,
    alpha=0.2,
    color="C2",
    label="Peroux & Howk (2020) Fit",
)
observation_lines.append(ax.plot(agrid, P20_rhoHI, color="C2")[0])
observation_labels.append("Peroux & Howk (2020) Fit")


W20_data = np.genfromtxt(
    f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"
    "/data/CosmicHIAbundance/raw/Walter2020_rhoHI.txt",
    usecols=[1, 2, 3],
)
W20eq2 = lambda z, a, b, c: (a * np.tanh(1 + z - b) + c)
W20_rhoHI = W20eq2(zgrid, *W20_data[0])
W20_rhoHI_lo = W20eq2(zgrid, *W20_data[1])
W20_rhoHI_hi = W20eq2(zgrid, *W20_data[2])

# Cosmology correction
# The assumed cosmology in Walter et al. (2020) has
# h=0.7, Omega_M = 0.31, Omega_Lambda = 0.69
W20_rhoHI *= cosmo.h ** -1 * np.sqrt(cosmo.Om0 * (1.0 + zgrid) ** 3 + cosmo.Ode0)
W20_rhoHI /= 0.7 ** -1 * np.sqrt(0.31 * (1.0 + zgrid) ** 3 + 0.69)
W20_rhoHI_lo *= cosmo.h ** -1 * np.sqrt(cosmo.Om0 * (1.0 + zgrid) ** 3 + cosmo.Ode0)
W20_rhoHI_lo /= 0.7 ** -1 * np.sqrt(0.31 * (1.0 + zgrid) ** 3 + 0.69)
W20_rhoHI_hi *= cosmo.h ** -1 * np.sqrt(cosmo.Om0 * (1.0 + zgrid) ** 3 + cosmo.Ode0)
W20_rhoHI_hi /= 0.7 ** -1 * np.sqrt(0.31 * (1.0 + zgrid) ** 3 + 0.69)

# convert from neutral gas density (H+He) to HI (only H) mass density
W20_rhoHI *= 0.76
W20_rhoHI_lo *= 0.76
W20_rhoHI_hi *= 0.76

ax.fill_between(
    pow(1 + zgrid, -1),
    W20_rhoHI_lo,
    W20_rhoHI_hi,
    alpha=0.2,
    color="C3",
    label="Walter et al. (2020) Fit",
)
observation_lines.append(ax.plot(pow(1 + zgrid, -1), W20_rhoHI, color="C3")[0])
observation_labels.append("Walter et al. (2020) Fit")

Firebox2022 = load_observations(
    [
        f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"
        "/data/CosmicHIAbundance/FIREbox.hdf5"
    ]
)[0]
observation_lines.append(
    ax.plot(
        Firebox2022.x.value,
        Firebox2022.y.value,
        color="black",
        zorder=-10000,
        dashes=(1.5, 1),
        alpha=0.7,
    )[0]
)
observation_labels.append(Firebox2022.citation)

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"Atomic Gas Cosmic Density $\rho_{\rm HI} [{\rm M_\odot \; cMpc^{-3}}]$")

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
ax.tick_params(axis="x", which="minor", bottom=False, top=False)

ax.set_xlim(1.02, 0.07)
ax.set_ylim(5e6, 1e10)

simulation_legend = ax.legend(
    simulation_lines, simulation_labels, markerfirst=False, loc="lower right"
)
observation_legend = ax.legend(
    observation_lines, observation_labels, markerfirst=True, loc="upper left"
)

ax.add_artist(simulation_legend)
ax.add_artist(observation_legend)

fig.savefig(f"{output_path}/HI_mass_evolution.png")
