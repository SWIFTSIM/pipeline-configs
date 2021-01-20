"""
Plots wallclock v.s. time-bin
"""

import unyt

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load
from swiftpipeline.argumentparser import ScriptArgumentParser
from glob import glob

arguments = ScriptArgumentParser(
    description="Creates a run performance plot: wall-clock time in each time-bin"
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

    timesteps_glob = glob(f"{run_directory}/timesteps_*.txt")
    timesteps_filename = timesteps_glob[0]
    snapshot_filename = f"{run_directory}/{snapshot_name}"

    snapshot = load(snapshot_filename)
    data = np.genfromtxt(
        timesteps_filename, skip_footer=5, loose=True, invalid_raise=False
    ).T

    time_bin_max = unyt.unyt_array(data[6], units="dimensionless")
    wallclock_time = unyt.unyt_array(data[-2], units="ms").to("Hour")

    time_bins = np.linspace(0, 57, 58)
    times = unyt.unyt_array(np.zeros(np.size(time_bins)), units="Hour")
    wallclock_time = wallclock_time[1:]
    time_bin_max = time_bin_max[1:]

    for i in range(np.size(time_bins)):
        times[i] = np.sum(wallclock_time[time_bin_max == time_bins[i]])

    # Simulation data plotting
    ax.plot(time_bins, times, zorder=10, label=run_name)

    ax.axvline(56, color="k", ls="--", lw=1)

ax.set_xlim(28, 58)

ax.legend(loc="upper left")

ax.set_xlabel("Time-bin [-]")
ax.set_ylabel("Wallclock time [Hours]")

fig.savefig(f"{output_path}/wallclock_timebin_hist.png")
