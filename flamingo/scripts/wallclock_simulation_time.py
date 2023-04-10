"""
Plots wallclock v.s. simulation time.
"""

import unyt

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load
from swiftpipeline.argumentparser import ScriptArgumentParser
from glob import glob

arguments = ScriptArgumentParser(
    description="Creates a run performance plot: simulation time versus wall-clock time"
)

run_names = arguments.name_list
run_directories = [f"{directory}" for directory in arguments.directory_list]
snapshot_names = [f"{snapshot}" for snapshot in arguments.snapshot_list]
output_path = arguments.output_directory

plt.style.use(arguments.stylesheet_location)

fig, ax = plt.subplots()

for run_name, run_directory, snapshot_name in zip(
    run_names, run_directories, snapshot_names
):

    timesteps_glob = glob(f"{run_directory}/timesteps*.txt")
    timesteps_filename = timesteps_glob[0]
    snapshot_filename = f"{run_directory}/{snapshot_name}"

    snapshot = load(snapshot_filename)
    data = np.genfromtxt(
        timesteps_filename,
        skip_footer=5,
        loose=True,
        invalid_raise=False,
        usecols=(1, 12),
        dtype=[("time", "f4"), ("wallclock", "f4")],
    )

    sim_time = unyt.unyt_array(data["time"], units=snapshot.units.time).to("Gyr")
    wallclock_time = unyt.unyt_array(np.cumsum(data["wallclock"]), units="ms").to(
        "Hour"
    )

    # Simulation data plotting
    (mpl_line,) = ax.plot(wallclock_time, sim_time, label=run_name)

    ax.scatter(
        wallclock_time[-1],
        sim_time[-1],
        color=mpl_line.get_color(),
        marker=".",
        zorder=10,
    )

ax.set_xlim(0, None)
ax.set_xlim(0, None)

ax.legend(loc="lower right")

ax.set_ylabel("Simulation time [Gyr]")
ax.set_xlabel("Wallclock time [Hours]")

fig.savefig(f"{output_path}/wallclock_simulation_time.png")
