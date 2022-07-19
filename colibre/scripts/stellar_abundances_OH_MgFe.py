"""
Plots the stellar abundances ([O/H] vs [Mg/Fe]) for a given snapshot
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
from unyt import unyt_array
from scipy import stats


def read_data(data):
    """
    Grabs the data
    """

    mH_in_cgs = unyt.mh
    mFe_in_cgs = 55.845 * unyt.mp
    mO_in_cgs = 15.999 * unyt.mp
    mMg_in_cgs = 24.305 * unyt.mp

    # Asplund et al. (2009)
    Fe_H_Sun_Asplund = 7.5
    Mg_H_Sun_Asplund = 7.6
    O_H_Sun_Asplund = 8.69

    Mg_Fe_Sun = Mg_H_Sun_Asplund - Fe_H_Sun_Asplund - np.log10(mFe_in_cgs / mMg_in_cgs)
    O_H_Sun = O_H_Sun_Asplund - 12.0 - np.log10(mH_in_cgs / mO_in_cgs)

    magnesium = data.stars.element_mass_fractions.magnesium
    iron = data.stars.element_mass_fractions.iron
    oxygen = data.stars.element_mass_fractions.oxygen
    hydrogen = data.stars.element_mass_fractions.hydrogen

    O_H = np.log10(oxygen / hydrogen) - O_H_Sun
    Mg_Fe = np.log10(magnesium / iron) - Mg_Fe_Sun

    O_H[oxygen == 0] = -4  # set lower limit
    O_H[O_H < -4] = -4  # set lower limit

    Mg_Fe[iron == 0] = -2  # set lower limit
    Mg_Fe[magnesium == 0] = -2  # set lower limit
    Mg_Fe[Mg_Fe < -2] = -2  # set lower limit

    return O_H, Mg_Fe


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

    O_H, Mg_Fe = read_data(data)

    # Bins along the X axis (O_H) to plot the median line
    bins = np.arange(-4.1, 1, 0.2)
    Min_N_points_per_bin = 11

    xm = 0.5 * (bins[1:] + bins[:-1])
    ym, _, _ = stats.binned_statistic(O_H, Mg_Fe, statistic="median", bins=bins)
    ym1, _, _ = stats.binned_statistic(
        O_H, Mg_Fe, statistic=lambda x: np.percentile(x, 16.0), bins=bins
    )
    ym2, _, _ = stats.binned_statistic(
        O_H, Mg_Fe, statistic=lambda x: np.percentile(x, 84.0), bins=bins
    )
    counts, _, _ = stats.binned_statistic(O_H, Mg_Fe, statistic="count", bins=bins)
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

# We select APOGEE data file
path_to_obs_data = f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"
observational_data = (
    f"{path_to_obs_data}/data/StellarAbundances/APOGEE_data_OHMGFE.hdf5"
)
x = unyt_array.from_hdf5(observational_data, dataset_name="values", group_name="x")
y = unyt_array.from_hdf5(observational_data, dataset_name="values", group_name="y")

xmin = -3
xmax = 2
ymin = -1
ymax = 2

ngridx = 100
ngridy = 50

# Create grid values first.
xi = np.linspace(xmin, xmax, ngridx)
yi = np.linspace(ymin, ymax, ngridy)

# Create a histogram
h, xedges, yedges = np.histogram2d(x.value, y.value, bins=(xi, yi))
xbins = xedges[:-1] + (xedges[1] - xedges[0]) / 2
ybins = yedges[:-1] + (yedges[1] - yedges[0]) / 2

z = h.T

binsize = 0.25
grid_min = np.log10(10)
grid_max = np.log10(np.ceil(h.max()))
levels = np.arange(grid_min, grid_max, binsize)
levels = 10 ** levels

contour = plt.contour(
    xbins, ybins, z, levels=levels, linewidths=0.5, cmap="winter", zorder=100
)

ax.set_xlabel("[O/H]")
ax.set_ylabel("[Mg/Fe]")

ax.set_ylim(-1.5, 1.5)
ax.set_xlim(-4.0, 2.0)

ax.annotate("APOGEE data", (-3.8, 1.3))

simulation_legend = ax.legend(
    simulation_lines, simulation_labels, markerfirst=False, loc="lower left"
)

ax.add_artist(simulation_legend)

plt.savefig(f"{output_path}/stellar_abundances_OH_MgFe.png")
