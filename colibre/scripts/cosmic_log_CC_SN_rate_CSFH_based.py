"""
Plots the star formation history. Modified version of the script in the
github.com/swiftsim/swiftsimio-examples repository.
"""
import unyt

import matplotlib.pyplot as plt
import numpy as np
import glob

from swiftsimio import load

from load_sfh_data import read_obs_data

from velociraptor.observations import load_observations

from astropy.cosmology import z_at_value
from astropy.units import Gyr

sfr_output_units = unyt.msun / (unyt.year * unyt.Mpc ** 3)
log_multiplicative_factor = 4
multiplicative_factor = 10 ** log_multiplicative_factor
SNIa_output_units = 1. / (unyt.year * unyt.Mpc ** 3)

from swiftpipeline.argumentparser import ScriptArgumentParser

def CC_SN_DTD(t, t_min, t_max):
    mask = (t > t_min) & (t < t_max)
    value = np.zeros(len(t)) / unyt.Gyr 
    value_new = 1./(t_max - t_min) * (np.ones(len(t[mask])))
    value_new.convert_to_units("Gyr**-1")
    value[mask] = value_new 
    return value

arguments = ScriptArgumentParser(
    description="Creates a CC SN history plot, with added observational data."
)

snapshot_filenames = [
    f"{directory}/{snapshot}"
    for directory, snapshot in zip(arguments.directory_list, arguments.snapshot_list)
]

sfr_filenames = [f"{directory}/SFR.txt" for directory in arguments.directory_list]

names = arguments.name_list
output_path = arguments.output_directory

plt.style.use(arguments.stylesheet_location)

simulation_lines = []
simulation_labels = []

fig, ax = plt.subplots()

ax.set_xscale("log")
ax.set_yscale("log")


for idx, (snapshot_filename, sfr_filename, name) in enumerate(
    zip(snapshot_filenames, sfr_filenames, names)
):
    data = np.genfromtxt(sfr_filename).T

    snapshot = load(snapshot_filename)

    # Read cosmology from the first run in the list
    if idx == 0:
        cosmology = snapshot.metadata.cosmology

    units = snapshot.units
    boxsize = snapshot.metadata.boxsize
    box_volume = boxsize[0] * boxsize[1] * boxsize[2]

    sfr_units = snapshot.gas.star_formation_rates.units

    # a, Redshift, SFR
    scale_factor = data[2]
    redshift = data[3]
    star_formation_rate = (data[7] * sfr_units / box_volume).to(sfr_output_units)
    times = data[1] * units.time 
    dM = data[4] * units.mass
    
    # Calculate the times we plot
    times_desired = np.linspace(0.05,13.8,500) * unyt.Gyr
    scale_factors_use = np.interp(times_desired, times, scale_factor)

    SNIa_rate = np.zeros(len(times_desired))

    for i in range(1,len(times_desired)):
        time_consider = times[times < times_desired[i]]
        time_since_formation = times_desired[i] - time_consider
        dM_consider = dM[times < times_desired[i]]
        SNIa_rate_individual = 1.180e-2 * CC_SN_DTD(time_since_formation, 3*unyt.Myr, 0.04*unyt.Gyr) * dM_consider / unyt.Msun

        SNIa_rate_sum = np.sum(SNIa_rate_individual)
        SNIa_rate[i] = SNIa_rate_sum

    SNIa_rate = (SNIa_rate / box_volume / unyt.Gyr).to(SNIa_output_units)

    # High z-order as we always want these to be on top of the observations
    simulation_lines.append(
        ax.plot(scale_factors_use, SNIa_rate.value * multiplicative_factor, zorder=10000)[0]
    )
    simulation_labels.append(name)

observation_lines = []
observation_labels = []

path_to_obs_data = f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"
observational_data = load_observations(
    sorted(glob.glob(f"{path_to_obs_data}/data/CosmicCCSNRate/*.hdf5"))
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
    observation_lines, observation_labels, markerfirst=True, loc=4, fontsize=4, ncol=2
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

ax.set_ylim(1e-1, 3e1)
ax.set_xlim(1.02, 0.07)
ax2.set_xlim(1.02, 0.07)

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(f"CC SN rate [$10^{{-{log_multiplicative_factor}}}$ yr$^{{-1}}$ cMpc$^{{-3}}$]")
ax2.set_xlabel("Cosmic time [Gyr]")

fig.savefig(f"{output_path}/log_CC_SN_rate_history_based_on_CSFH.png")
