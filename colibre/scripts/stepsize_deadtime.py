"""
Plots dead time fraction v.s. step size
"""

import matplotlib.pyplot as plt
import numpy as np

from swiftpipeline.argumentparser import ScriptArgumentParser
from glob import glob
import re

arguments = ScriptArgumentParser(
    description="Creates a run performance plot: dead time fraction vs step size"
)

run_names = arguments.name_list
run_directories = [f"{directory}" for directory in arguments.directory_list]
output_path = arguments.output_directory

plt.style.use(arguments.stylesheet_location)

fig, ax = plt.subplots()

color_index = 0
for run_name, run_directory in zip(run_names, run_directories):

    color = f"C{color_index}"
    color_index += 1

    timesteps_glob = glob(f"{run_directory}/timesteps*.txt")
    timesteps_filename = timesteps_glob[0]

    # extract the number of ranks from the file
    nrank = 1
    with open(timesteps_filename, "r") as file:
        header = file.readlines()[:10]
        for line in header:
            match = re.findall("Number of MPI ranks: (\d+)", line)
            if len(match) > 0:
                nrank = int(match[0])
                break
    try:
        data = np.genfromtxt(
            timesteps_filename,
            skip_footer=5,
            loose=True,
            invalid_raise=True,
            usecols=(8, 12, 13, 14),
            dtype=[
                ("updates", "i8"),
                ("wallclock", "f4"),
                ("properties", "i4"),
                ("deadtime", "f4"),
            ],
        )
    except:
        ax.plot([], [], "-", color=color, label=f"{run_name} - no deadtime data")
        continue

    # filter out all steps where something special happened
    data = data[data["properties"] == 0]

    nupdates = data["updates"] / nrank
    deadtime = data["deadtime"] / data["wallclock"]

    # bin the data
    nbins = 10.0 ** np.linspace(np.log10(nupdates.min()), np.log10(nupdates.max()), 51)
    # make sure the highest values is inside the top bin
    nbins[-1] += 0.0001 * (nbins[-1] - nbins[0])
    means = np.zeros(50)
    keep = np.array([], dtype=np.int32)
    for ibin in range(50):
        sel = np.nonzero((nupdates >= nbins[ibin]) & (nupdates < nbins[ibin + 1]))[0]
        # do not bin values if there are less than 5 in the bin
        if sel.shape[0] < 5:
            means[ibin] = np.nan
            keep = np.append(keep, sel)
        else:
            means[ibin] = np.mean(deadtime[sel])

    nbins = 0.5 * (nbins[1:] + nbins[:-1])

    # Simulation data plotting
    # first plot the values that were not part of any bin
    ax.semilogx(nupdates[keep], deadtime[keep], ".", color=color)
    # now add the binned values
    ax.semilogx(nbins, means, "-", color=color, label=f"{run_name} [{nrank} ranks]")

ax.legend(loc="upper left")

ax.set_xlim(1.0e-3, 1e9)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel("Number of updates per rank")
ax.set_ylabel("Deadtime fraction")

fig.savefig(f"{output_path}/stepsize_deadtime.png")
