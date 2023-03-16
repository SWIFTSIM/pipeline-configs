"""
Plots the cosmic r_process rate history.
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
    description="Creates a r-process rate history plot, with added observational data."
)

snapshot_filenames = [
    f"{directory}/{snapshot}"
    for directory, snapshot in zip(arguments.directory_list, arguments.snapshot_list)
]

r_process_filenames = [f"{directory}/r_processes.txt" for directory in arguments.directory_list]

names = arguments.name_list
output_path = arguments.output_directory

plt.style.use(arguments.stylesheet_location)

simulation_lines = []
simulation_labels = []

fig, ax = plt.subplots()

ax.semilogx()

log_multiplicative_factor = 4
multiplicative_factor = 10 ** log_multiplicative_factor
r_process_rate_output_units = 1.0 / (unyt.yr * unyt.Mpc ** 3)

for idx, (snapshot_filename, r_process_filename, name) in enumerate(
    zip(snapshot_filenames, r_process_filenames, names)
):
    data = np.loadtxt(
        r_process_filename,
        usecols=(2,3, 4, 6, 14, 15, 16),
        dtype=[("t2", np.float32), ("t1", np.float32), ("a", np.float32), ("z", np.float32), ("NSM", np.float32), ("CEJSN", np.float32), ("collapsar", np.float32)],
    )

    snapshot = load(snapshot_filename)

    # Read cosmology from the first run in the list
    if idx == 0:
        cosmology = snapshot.metadata.cosmology

    units = snapshot.units
    r_process_rate_units = 1.0 / (units.time * units.length ** 3)

    volume = snapshot.metadata.boxsize[0] * snapshot.metadata.boxsize[1]* snapshot.metadata.boxsize[2]
    volume.convert_to_units("Mpc**3")

    dt = (data["t2"] - data["t1"]) 

    # a, Redshift, SFR
    scale_factor = data["a"]
    NSM_rate = (data["NSM"]/volume.value /dt * r_process_rate_units).to(r_process_rate_output_units)
    CEJSN_rate = (data["CEJSN"]/volume.value/dt * r_process_rate_units).to(r_process_rate_output_units)
    collapsar_rate = (data["collapsar"]/volume.value/dt * r_process_rate_units).to(r_process_rate_output_units)

    # High z-order as we always want these to be on top of the observations
    simulation_lines.append(
        ax.plot(scale_factor, collapsar_rate.value * multiplicative_factor, zorder=10000)[0]
    )
    simulation_labels.append(name)


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

simulation_legend = ax.legend(
    simulation_lines, simulation_labels, markerfirst=False, loc="upper right"
)

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

ax.set_ylim(1e-5, 2.0)
ax.set_xlim(1.02, 0.07)
ax2.set_xlim(1.02, 0.07)

ax.set_xlabel("Redshift $z$")
ax.set_ylabel(
    f"Collapsar rate [$10^{{-{log_multiplicative_factor}}}$ yr$^{{-1}}$ cMpc$^{{-3}}$]"
)
ax2.set_xlabel("Cosmic time [Gyr]")

fig.savefig(f"{output_path}/log_collapsar_rate_history.png")
