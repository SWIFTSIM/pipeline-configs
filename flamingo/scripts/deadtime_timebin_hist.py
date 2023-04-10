"""
Plots dead time fraction v.s. time-bin
"""

import matplotlib.pyplot as plt
import numpy as np

from swiftpipeline.argumentparser import ScriptArgumentParser
from glob import glob

arguments = ScriptArgumentParser(
    description="Creates a run performance plot: deadtime fraction in each time-bin"
)

run_names = arguments.name_list
run_directories = [f"{directory}" for directory in arguments.directory_list]
output_path = arguments.output_directory

plt.style.use(arguments.stylesheet_location)

fig, ax = plt.subplots()

for run_name, run_directory in zip(run_names, run_directories):

    timesteps_glob = glob(f"{run_directory}/timesteps*.txt")
    timesteps_filename = timesteps_glob[0]

    try:
        data = np.genfromtxt(
            timesteps_filename,
            skip_footer=5,
            loose=True,
            invalid_raise=True,
            usecols=(6, 12, 14),
            dtype=[("time_bin", "i1"), ("wallclock", "f4"), ("deadtime", "f4")],
        )
    except:
        ax.step([], [], label=f"{run_name} - no deadtime data")
        continue

    time_bin_max = data["time_bin"]
    wallclock_time = data["wallclock"]
    dead_time = data["deadtime"]

    time_bins = np.linspace(0, 57, 58)
    wallclock_times = np.zeros(time_bins.shape)
    dead_times = np.zeros(time_bins.shape)
    time_bin_max = time_bin_max[1:]
    wallclock_time = wallclock_time[1:]
    dead_time = dead_time[1:]

    for i in range(np.size(time_bins)):
        wallclock_times[i] = np.sum(wallclock_time[time_bin_max == time_bins[i]])
        dead_times[i] = np.sum(dead_time[time_bin_max == time_bins[i]])

    fractions = np.zeros(dead_times.shape)
    fractions[wallclock_times > 0] = (
        dead_times[wallclock_times > 0] / wallclock_times[wallclock_times > 0]
    )

    # Simulation data plotting
    ax.step(time_bins + 0.5, fractions, zorder=10, label=run_name)

ax.axvline(56, color="k", ls="--", lw=1)

ax.set_xlim(28, 57)

ax.legend(loc="upper left")

ax.set_xlabel("Time-bin [-]")
ax.set_ylabel("Deadtime fraction")

fig.savefig(f"{output_path}/deadtime_timebin_hist.png")
