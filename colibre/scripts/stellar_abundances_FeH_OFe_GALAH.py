"""
Plots the stellar abundances ([Fe/H] vs [O/Fe]) for a given snapshot
"""
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mc
import numpy as np
import unyt
import glob
from swiftsimio import load
from swiftpipeline.argumentparser import ScriptArgumentParser
from velociraptor.observations import load_observations
import h5py

def read_data(data):
    """
    Grabs the data
    """

    mH_in_cgs = unyt.mh
    mFe_in_cgs = 55.845 * unyt.mp
    mO_in_cgs = 15.999 * unyt.mp

    # Asplund et al. (2009)
    Fe_H_Sun_Asplund = 7.5
    O_H_Sun_Asplund = 8.69

    O_Fe_Sun = O_H_Sun_Asplund - Fe_H_Sun_Asplund - np.log10(mFe_in_cgs / mO_in_cgs)
    Fe_H_Sun = Fe_H_Sun_Asplund - 12.0 - np.log10(mH_in_cgs / mFe_in_cgs)

    oxygen = data.stars.element_mass_fractions.oxygen
    iron = data.stars.element_mass_fractions.iron
    hydrogen = data.stars.element_mass_fractions.hydrogen

    Fe_H = np.log10(iron / hydrogen) - Fe_H_Sun
    O_Fe = np.log10(oxygen / iron) - O_Fe_Sun

    Fe_H[iron == 0] = -4  # set lower limit
    Fe_H[Fe_H < -4] = -4  # set lower limit

    O_Fe[iron == 0] = -2  # set lower limit
    O_Fe[oxygen == 0] = -2  # set lower limit
    O_Fe[O_Fe < -2] = -2  # set lower limit

    return Fe_H, O_Fe


arguments = ScriptArgumentParser(
    description="Creates an [Fe/H] - [O/Fe] plot for stars."
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

    data = load(snapshot_filename)
    redshift = data.metadata.z

    Fe_H, O_Fe = read_data(data)

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
            ym.append(np.median(O_Fe[in_bin_idx]))
            ym1.append(np.percentile(O_Fe[in_bin_idx], 16))
            ym2.append(np.percentile(O_Fe[in_bin_idx], 84))

    fill_element = ax.fill_between(xm, ym1, ym2, alpha=0.2)

    # high zorder, as we want the simulation lines to be on top of everything else
    # we steal the color of the dots to make sure the line has the same color
    simulation_lines.append(
        ax.plot(
            xm,
            ym,
            lw=2,
            color=mc.to_hex(fill_element.get_facecolor()[0], keep_alpha = False),
            zorder=1000,
            path_effects=[pe.Stroke(linewidth=4, foreground="white"), pe.Normal()],
        )[0]
    )
    simulation_labels.append(f"{name} ($z={redshift:.1f}$)")

# We select all GALAH file containing FeH-MgFe.
path_to_obs_data = f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"
observational_data = f"{path_to_obs_data}/data/StellarAbundances/Buder21_data.hdf5"

GALAH_data = h5py.File(observational_data, 'r')
galah_edges = np.array(GALAH_data["abundance_bin_edges"])
obs_plane = np.array(GALAH_data["Mg_enrichment_vs_Fe_abundance"]).T
obs_plane[obs_plane < 10] = None

contour = plt.contour(np.log10(obs_plane), origin='lower',
                      extent=[galah_edges[0], galah_edges[-1],
                              galah_edges[0], galah_edges[-1]],
                      zorder=100, cmap='winter', linewidths=0.5)

ax.set_xlabel("[Fe/H]")
ax.set_ylabel("[O/Fe]")

ax.set_ylim(-1.5, 1.5)
ax.set_xlim(-4.0, 2.0)

ax.annotate('GALAH data',(-3.8,1.3))

simulation_legend = ax.legend(
    simulation_lines, simulation_labels, markerfirst=False, loc="lower left"
)

ax.add_artist(simulation_legend)

plt.savefig(f"{output_path}/stellar_abundances_FeH_OFe_GALAH.png")
