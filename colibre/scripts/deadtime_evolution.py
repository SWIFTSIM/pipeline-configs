"""
Plots dead time fraction v.s. cosmic scale factor
"""

import matplotlib.pyplot as plt
import numpy as np

from swiftpipeline.argumentparser import ScriptArgumentParser
from glob import glob
import re
import scipy.stats as stats

arguments = ScriptArgumentParser(
    description="Creates a run performance plot: evolution of the dead time fraction in scale factor bins"
)

run_names = arguments.name_list
run_directories = [f"{directory}" for directory in arguments.directory_list]
output_path = arguments.output_directory

plt.style.use(arguments.stylesheet_location)

fig, ax = plt.subplots()

for color_index, (run_name, run_directory) in enumerate(
    zip(run_names, run_directories)
):

    color = f"C{color_index}"

    timesteps_glob = glob(f"{run_directory}/timesteps_*.txt")
    timesteps_filename = timesteps_glob[0]

    # extract the number of ranks from the file
    nrank = 1
    nthread = 1
    with open(timesteps_filename, "r") as file:
        header = file.readlines()[:10]
        for line in header:
            match = re.findall("Number of MPI ranks: (\d+)", line)
            if len(match) > 0:
                nrank = int(match[0])
            match = re.findall("Number of threads: (\d+)", line)
            if len(match) > 0:
                nthread = int(match[0])
    try:
        data = np.genfromtxt(
            timesteps_filename,
            skip_footer=5,
            loose=False,
            invalid_raise=True,
            usecols=(2, 12, 14),
            dtype=[
                ("a", "f4"),
                ("wallclock", "f4"),
                ("deadtime", "f4"),
            ],
        )
    except (FileNotFoundError, ValueError):
        ax.plot([], [], "-", color=color, label=f"{run_name} - no deadtime data")
        continue

    a = data["a"]
    deadtime = data["deadtime"]
    walltime = data["wallclock"]

    # bin the data
    abins = np.linspace(a.min(), a.max(), 51)
    deadbins, _, _ = stats.binned_statistic(a, deadtime, statistic="sum", bins=abins)
    wallbins, _, _ = stats.binned_statistic(a, walltime, statistic="sum", bins=abins)
    abins = 0.5 * (abins[1:] + abins[:-1])

    ax.plot(
        abins,
        deadbins / wallbins,
        "-",
        color=color,
        label=f"{run_name} [{nrank} ranks, {nrank*nthread} cores]",
    )
    # also plot the mean deadtime for the whole run
    ax.axhline(y=deadtime.sum() / walltime.sum(), linestyle="--", color=color)

ax.legend(loc="upper left")

ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel("Scale factor")
ax.set_ylabel("Deadtime fraction")

fig.savefig(f"{output_path}/deadtime_evolution.png")
