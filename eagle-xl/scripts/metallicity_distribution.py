"""
Plots the metal mass fraction distribution for stars and gas.
"""

import matplotlib.pyplot as plt
import numpy as np
import unyt

from swiftsimio import load

from unyt import mh, cm, Gyr
from matplotlib.colors import LogNorm, Normalize
from matplotlib.animation import FuncAnimation

from swiftpipeline.argumentparser import ScriptArgumentParser

arguments = ScriptArgumentParser(
    description="Creates metal mass fraction distribution plots for stars and gas."
)

snapshot_filenames = [
    f"{directory}/{snapshot}"
    for directory, snapshot in zip(arguments.directory_list, arguments.snapshot_list)
]

names = arguments.name_list
output_path = arguments.output_directory

plt.style.use(arguments.stylesheet_location)

data = [load(snapshot_filename) for snapshot_filename in snapshot_filenames]
number_of_bins = 256

metallicity_bins = np.logspace(-10, 0, number_of_bins)
metallicity_bin_centers = 0.5 * (metallicity_bins[1:] + metallicity_bins[:-1])
log_metallicity_bin_width = np.log10(metallicity_bins[1]) - np.log10(
    metallicity_bins[0]
)


# Begin plotting

fig, ax = plt.subplots()

ax.loglog()

for color, (snapshot, name) in enumerate(zip(data, names)):
    try:
        metallicities = {
            "Gas": np.histogram(
                snapshot.gas.smoothed_metal_mass_fractions.value, bins=metallicity_bins
            )[0],
            "Stars": np.histogram(
                snapshot.stars.smoothed_metal_mass_fractions.value, bins=metallicity_bins
            )[0],
        }
        smoothed = True
    except AttributeError:
        metallicities = {
            "Gas": np.histogram(
                snapshot.gas.metal_mass_fractions.value, bins=metallicity_bins
            )[0],
            "Stars": np.histogram(
                snapshot.stars.metal_mass_fractions.value, bins=metallicity_bins
            )[0],
        }
        smoothed = False

    ax.plot(
        metallicity_bin_centers,
        metallicities["Gas"] / log_metallicity_bin_width,
        label=name,
        color=f"C{color}",
        linestyle="solid",
    )

    ax.plot(
        metallicity_bin_centers,
        metallicities["Stars"] / log_metallicity_bin_width,
        color=f"C{color}",
        linestyle="dashed",
    )


simulation_legend = ax.legend(loc="upper right", markerfirst=False)

from matplotlib.lines import Line2D

custom_lines = [
    Line2D([0], [0], color="black", linestyle="solid"),
    Line2D([0], [0], color="black", linestyle="dashed"),
]
ax.legend(custom_lines, ["Gas", "Stars"], markerfirst=True, loc="upper left")
ax.add_artist(simulation_legend)
ax.set_xlabel(f"{'Smoothed ' if smoothed else ''}Metal Mass Fractions $Z$ []")
ax.set_ylabel("Number of Particles / d$\\log Z$")

fig.savefig(f"{output_path}/metallicity_distribution.png")
