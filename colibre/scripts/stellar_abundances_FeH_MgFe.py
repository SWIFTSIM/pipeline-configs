"""
Plots the stellar abundances ([Fe/H] vs [Mg/Fe]) for a given snapshot
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
from scipy import stats


def read_data(data):
    """
    Grabs the data
    """

    mH_in_cgs = unyt.mh
    mFe_in_cgs = 55.845 * unyt.mp
    mMg_in_cgs = 24.305 * unyt.mp

    # Asplund et al. (2009)
    Fe_H_Sun_Asplund = 7.5
    Mg_H_Sun_Asplund = 7.6

    Mg_Fe_Sun = Mg_H_Sun_Asplund - Fe_H_Sun_Asplund - np.log10(mFe_in_cgs / mMg_in_cgs)
    Fe_H_Sun = Fe_H_Sun_Asplund - 12.0 - np.log10(mH_in_cgs / mFe_in_cgs)

    magnesium = data.stars.element_mass_fractions.magnesium
    iron = data.stars.element_mass_fractions.iron
    hydrogen = data.stars.element_mass_fractions.hydrogen

    Fe_H = np.log10(iron / hydrogen) - Fe_H_Sun
    Mg_Fe = np.log10(magnesium / iron) - Mg_Fe_Sun

    Fe_H[iron == 0] = -4  # set lower limit
    Fe_H[Fe_H < -4] = -4  # set lower limit

    Mg_Fe[iron == 0] = -2  # set lower limit
    Mg_Fe[magnesium == 0] = -2  # set lower limit
    Mg_Fe[Mg_Fe < -2] = -2  # set lower limit

    return Fe_H, Mg_Fe


arguments = ScriptArgumentParser(
    description="Creates an [Fe/H] - [Mg/Fe] plot for stars."
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

    Fe_H, Mg_Fe = read_data(data)

    # Bins along the X axis (Fe_H) to plot the median line
    bins = np.arange(-4.1, 1, 0.2)
    Min_N_points_per_bin = 11

    xm = 0.5 * (bins[1:] + bins[:-1])
    ym, _, _ = stats.binned_statistic(Fe_H, Mg_Fe, statistic="median", bins=bins)
    ym1, _, _ = stats.binned_statistic(
        Fe_H, Mg_Fe, statistic=lambda x: np.percentile(x, 16.0), bins=bins
    )
    ym2, _, _ = stats.binned_statistic(
        Fe_H, Mg_Fe, statistic=lambda x: np.percentile(x, 84.0), bins=bins
    )
    counts, _, _ = stats.binned_statistic(Fe_H, Mg_Fe, statistic="count", bins=bins)
    mask = counts >= Min_N_points_per_bin
    xm = xm[mask]
    ym = ym[mask]
    ym1 = ym1[mask]
    ym2 = ym2[mask]

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

# We select all Tolstoy* files containing FeH-MgFe.
path_to_obs_data = f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"
observational_data = glob.glob(
    f"{path_to_obs_data}/data/StellarAbundances/Tolstoy*.hdf5"
)

for obs in load_observations(observational_data):
    obs.plot_on_axes(ax)

ax.set_xlabel("[Fe/H]")
ax.set_ylabel("[Mg/Fe]")

ax.set_ylim(-1.5, 2.0)
ax.set_xlim(-4, 2.0)

observation_legend = ax.legend(markerfirst=True, loc="upper left")

ax.add_artist(observation_legend)

simulation_legend = ax.legend(
    simulation_lines, simulation_labels, markerfirst=False, loc="lower left"
)

ax.add_artist(simulation_legend)

plt.savefig(f"{output_path}/stellar_abundances_FeH_MgFe.png")
