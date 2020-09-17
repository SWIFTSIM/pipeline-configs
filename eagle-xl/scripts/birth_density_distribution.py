"""
Plots the birth density distribution for stars.
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
    description="Creates a birth density distribution plot"
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

birth_density_bins = unyt.unyt_array(
    np.logspace(-3, 5, number_of_bins), units=1 / cm ** 3
)
log_birth_density_bin_width = np.log10(birth_density_bins[1].value) - np.log10(
    birth_density_bins[0].value
)
birth_density_centers = 0.5 * (birth_density_bins[1:] + birth_density_bins[:-1])

# Begin plotting

fig, ax = plt.subplots()

ax.loglog()

for color, (snapshot, name) in enumerate(zip(data, names)):

    birth_densities = (snapshot.stars.birth_densities / mh).to(birth_density_bins.units)
    birth_redshifts = 1 / snapshot.stars.birth_scale_factors.value - 1

    # Segment birth densities into redshift bins
    birth_densities_by_redshift = {
        "$z < 1$": birth_densities[birth_redshifts < 1],
        "$1 < z < 3$": birth_densities[
            np.logical_and(birth_redshifts > 1, birth_redshifts < 3)
        ],
        "$z > 3$": birth_densities[birth_redshifts > 3],
    }




    for index, (label, data) in enumerate(birth_densities_by_redshift.items()):
        if len(data) < 1:
            continue

        H, _ = np.histogram(data, bins=birth_density_bins)
        ax.plot(
            birth_density_centers,
            H / log_birth_density_bin_width,
            label=label,
            color=f"C{index}",
        )
        ax.axvline(
            np.median(data),
            color=f"C{index}",
            linestyle="dashed",
            zorder=-10,
            alpha=0.5,
        )

simulation_legend = ax.legend(loc="upper right", markerfirst=False)

ax.add_artist(simulation_legend)
ax.set_xlabel("Stellar Birth Density $\\rho_B$ [$n_H$ cm$^{-3}$]")
ax.set_ylabel("Number of Stars / d$\\log\\rho_B$")

fig.savefig(f"{output_path}/birth_density_distribution.png")
