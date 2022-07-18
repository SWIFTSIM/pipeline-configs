"""
Plots [Fe/H] vs mass fraction of Fe from SNIa
"""
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mc
import numpy as np
import swiftsimio
import unyt
from swiftsimio import load
from swiftpipeline.argumentparser import ScriptArgumentParser
from typing import Tuple


def read_data(data: swiftsimio.SWIFTDataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Grabs the data
    """

    mH_in_cgs = unyt.mh
    mFe_in_cgs = 55.845 * unyt.mp

    # Asplund et al. (2009)
    Fe_H_Sun_Asplund = 7.5

    Fe_H_Sun = Fe_H_Sun_Asplund - 12.0 - np.log10(mH_in_cgs / mFe_in_cgs)

    iron = data.stars.element_mass_fractions.iron
    iron_snia = data.stars.iron_mass_fractions_from_snia
    hydrogen = data.stars.element_mass_fractions.hydrogen

    Fe_H = np.log10(iron / hydrogen) - Fe_H_Sun
    Fe_snia_fraction = iron_snia / iron

    Fe_H[iron == 0] = -4  # set lower limit
    Fe_H[Fe_H < -4] = -4  # set lower limit

    return Fe_H, Fe_snia_fraction


arguments = ScriptArgumentParser(
    description="Creates an [Fe/H] - mass fraction of Fe from SNIa for stars."
)

snapshot_filenames = [
    f"{directory}/{snapshot}"
    for directory, snapshot in zip(arguments.directory_list, arguments.snapshot_list)
]

names = arguments.name_list
output_path = arguments.output_directory

plt.style.use(arguments.stylesheet_location)

simulation_lines = []
simulation_labels = []

fig, ax = plt.subplots()
ax.grid(True)

for snapshot_filename, name in zip(snapshot_filenames, names):

    snapshot_data = load(snapshot_filename)
    redshift = snapshot_data.metadata.z

    Fe_H, Fe_snia_fr = read_data(snapshot_data)

    # Bins along the X axis (Fe_H) to plot the median line
    bins = np.arange(-4.1, 1, 0.2)
    ind = np.digitize(Fe_H, bins)

    xm, ym, ym1, ym2 = [], [], [], []
    Min_N_points_per_bin = 11

    for i in range(1, len(bins)):
        in_bin_idx = ind == i
        N_data_points_per_bin = np.sum(in_bin_idx)
        if N_data_points_per_bin >= Min_N_points_per_bin:
            xm.append(np.median(Fe_H[in_bin_idx]))
            ym.append(np.median(Fe_snia_fr[in_bin_idx]))
            ym1.append(np.percentile(Fe_snia_fr[in_bin_idx], 16))
            ym2.append(np.percentile(Fe_snia_fr[in_bin_idx], 84))

    fill_element = ax.fill_between(xm, ym1, ym2, alpha=0.2)

    # high zorder, as we want the simulation lines to be on top of everything else
    # we steal the color of the dots to make sure the line has the same color
    simulation_lines.append(
        ax.plot(
            xm,
            ym,
            lw=2,
            color=mc.to_hex(fill_element.get_facecolor()[0], keep_alpha=False),
            zorder=1000,
            path_effects=[pe.Stroke(linewidth=4, foreground="white"), pe.Normal()],
        )[0]
    )
    simulation_labels.append(f"{name} ($z={redshift:.1f}$)")

ax.set_xlabel("[Fe/H]")
ax.set_ylabel("Fe (SNIa) / Fe (Total)")

ax.set_ylim(3e-3, 3.0)
ax.set_xlim(-4.0, 2.0)
ax.set_yscale("log")

simulation_legend = ax.legend(
    simulation_lines, simulation_labels, markerfirst=False, loc="lower left"
)

ax.add_artist(simulation_legend)

plt.savefig(f"{output_path}/stellar_abundances_FeH_Fe_snia_fraction.png")
