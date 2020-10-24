"""
Plots simulation time v.s. number of steps.
"""

import unyt

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load
from swiftpipeline.argumentparser import ScriptArgumentParser
from glob import glob

# Import EAGLE cosmology object
from astropy.cosmology import Planck13

arguments = ScriptArgumentParser(
    description="Creates a run performance plot: number of steps versus simulation time"
)

run_names = arguments.name_list
run_directories = [f"{directory}" for directory in arguments.directory_list]
snapshot_names = [f"{snapshot}" for snapshot in arguments.snapshot_list]
output_path = arguments.output_directory

plt.style.use(arguments.stylesheet_location)

fig, ax = plt.subplots()

# We will need to keep track of the maximum cosmic time reached in the simulation(s)
t_max = unyt.unyt_array(0., units="Gyr")

for run_name, run_directory, snapshot_name in zip(
    run_names, run_directories, snapshot_names
):

    timesteps_glob = glob(f"{run_directory}/timesteps_*.txt")
    timesteps_filename = timesteps_glob[0]
    snapshot_filename = f"{run_directory}/{snapshot_name}"

    snapshot = load(snapshot_filename)
    data = np.genfromtxt(
        timesteps_filename, skip_footer=5, loose=True, invalid_raise=False
    ).T

    sim_time = unyt.unyt_array(data[1], units=snapshot.units.time).to("Gyr")

    # Update the maximum cosmic time if needed
    if sim_time[-1] > t_max:
        t_max = sim_time[-1]

    number_of_steps = np.arange(sim_time.size) / 1e6

    # Simulation data plotting
    (mpl_line,) = ax.plot(sim_time, number_of_steps, label=run_name)

    ax.scatter(
        sim_time[-1],
        number_of_steps[-1],
        color=mpl_line.get_color(),
        marker=".",
        zorder=10,
    )
# Create second X-axis (to plot redshift alongside the cosmic time)
ax2 = ax.twiny()

# z ticks along the second X-axis we want to display
z_ticks = np.array([0, 0.2, 0.5, 1, 1.5, 2, 3, 10])

# To place the new ticks onto the X-axis we need to know the corresponding cosmic times
t_ticks = Planck13.age(z_ticks).value

# Attach the ticks to the second X-axis
ax2.set_xticks(t_ticks)

# Format the ticks' labels
ax2.set_xticklabels(["$%2.1f$" % z_tick for z_tick in z_ticks])

ax.set_xlim(0, t_max * 1.05)
ax2.set_xlim(0, t_max * 1.05)
ax.set_ylim(0, None)

ax.legend(loc="lower right")

ax.set_ylabel("Number of steps [millions]")
ax.set_xlabel("Simulation time [Gyr]")
ax2.set_xlabel("Redshift z")

fig.savefig(f"{output_path}/simulation_time_number_of_steps.png")
