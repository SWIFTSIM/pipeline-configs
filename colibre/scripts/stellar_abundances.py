"""
Plots the stellar abundances for a given snapshot
"""
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import unyt
import glob
from swiftsimio import load
from swiftpipeline.argumentparser import ScriptArgumentParser
from velociraptor.observations import load_observations
from unyt import unyt_array
from scipy import stats
import h5py


def make_hist(x, y, cut, xi, yi):

    selection = np.where(cut)[0]

    # Create a histogram
    h, xedges, yedges = np.histogram2d(
        x[selection], y[selection], bins=(xi, yi), density=True
    )

    return h, xedges, yedges


def make_stellar_abundance_distribution(x, y, R, z, xi, yi):

    # Galactocentric radius (R) in kpc units
    # Galactocentric azimuthal distance (z) in kpc units
    h = np.zeros((len(xi) - 1, len(yi) - 1))

    # We apply masks to select stars in R & z bins
    for Ri in range(0, 9, 3):
        for zi in range(0, 2, 1):
            mask_R = (R >= Ri) & (R < Ri + 3)
            mask_z = (np.abs(z) >= zi) & (np.abs(z) < zi + 1)
            distance_cut = np.logical_and(mask_R, mask_z)

            hist, xedges, yedges = make_hist(x, y, distance_cut, xi, yi)

            # We combine (add) the 6 histograms to give less weight to stars in the solar vicinity.
            # In this manner all stars in the radial/azimuthal bins contribute with equal weight to
            # the final stellar distribution
            h += hist

    return h, xedges, yedges


def read_data(data, xvar, yvar):
    """
    Grabs the data
    """

    mH_in_cgs = unyt.mh
    mC_in_cgs = 12.0107 * unyt.mp
    mN_in_cgs = 14.0067 * unyt.mp
    mO_in_cgs = 15.999 * unyt.mp
    mMg_in_cgs = 24.305 * unyt.mp
    mFe_in_cgs = 55.845 * unyt.mp

    mSi_in_cgs = 28.0855 * unyt.mp
    mEu_in_cgs = 151.964 * unyt.mp
    mBa_in_cgs = 137.327 * unyt.mp
    mSr_in_cgs = 87.62 * unyt.mp
    mN_in_cgs = 14.0067 * unyt.mp
    mNe_in_cgs = 20.1797 * unyt.mp

    # Asplund et al. (2009)
    C_H_Sun_Asplund = 8.43
    N_H_Sun_Asplund = 7.83
    O_H_Sun_Asplund = 8.69
    Mg_H_Sun_Asplund = 7.6
    Fe_H_Sun_Asplund = 7.5

    Si_H_Sun_Asplund = 7.51
    Eu_H_Sun_Asplund = 0.52
    Ba_H_Sun_Asplund = 2.18
    Sr_H_Sun_Asplund = 2.87
    Ne_H_Sun_Asplund = 7.93

    O_H_Sun = O_H_Sun_Asplund - 12.0 - np.log10(mH_in_cgs / mO_in_cgs)
    Fe_H_Sun = Fe_H_Sun_Asplund - 12.0 - np.log10(mH_in_cgs / mFe_in_cgs)

    C_Fe_Sun = C_H_Sun_Asplund - Fe_H_Sun_Asplund - np.log10(mFe_in_cgs / mC_in_cgs)
    N_Fe_Sun = N_H_Sun_Asplund - Fe_H_Sun_Asplund - np.log10(mFe_in_cgs / mN_in_cgs)
    O_Fe_Sun = O_H_Sun_Asplund - Fe_H_Sun_Asplund - np.log10(mFe_in_cgs / mO_in_cgs)
    N_O_Sun = N_H_Sun_Asplund - O_H_Sun_Asplund - np.log10(mO_in_cgs / mN_in_cgs)
    C_O_Sun = C_H_Sun_Asplund - O_H_Sun_Asplund - np.log10(mO_in_cgs / mC_in_cgs)
    Mg_Fe_Sun = Mg_H_Sun_Asplund - Fe_H_Sun_Asplund - np.log10(mFe_in_cgs / mMg_in_cgs)

    Si_Fe_Sun = Si_H_Sun_Asplund - Fe_H_Sun_Asplund - np.log10(mFe_in_cgs / mSi_in_cgs)
    Eu_Fe_Sun = Eu_H_Sun_Asplund - Fe_H_Sun_Asplund - np.log10(mFe_in_cgs / mEu_in_cgs)
    Ba_Fe_Sun = Ba_H_Sun_Asplund - Fe_H_Sun_Asplund - np.log10(mFe_in_cgs / mBa_in_cgs)
    Sr_Fe_Sun = Sr_H_Sun_Asplund - Fe_H_Sun_Asplund - np.log10(mFe_in_cgs / mSr_in_cgs)
    Ne_Fe_Sun = Ne_H_Sun_Asplund - Fe_H_Sun_Asplund - np.log10(mFe_in_cgs / mNe_in_cgs)

    hydrogen = data.stars.element_mass_fractions.hydrogen
    iron = data.stars.element_mass_fractions.iron

    if xvar == "O_H" or yvar == "O_Fe":
        oxygen = data.stars.element_mass_fractions.oxygen
    if yvar == "C_Fe":
        carbon = data.stars.element_mass_fractions.carbon
    if yvar == "N_Fe":
        nitrogen = data.stars.element_mass_fractions.nitrogen
    if yvar == "N_O":
        nitrogen = data.stars.element_mass_fractions.nitrogen
        oxygen = data.stars.element_mass_fractions.oxygen
    if yvar == "C_O":
        carbon = data.stars.element_mass_fractions.carbon
        oxygen = data.stars.element_mass_fractions.oxygen
    if yvar == "Mg_Fe":
        magnesium = data.stars.element_mass_fractions.magnesium
    if yvar == "Fe_SNIa_fraction":
        iron_snia = data.stars.iron_mass_fractions_from_snia
    if yvar == "Si_Fe":
        silicon = data.stars.element_mass_fractions.silicon
    if yvar == "Ne_Fe":
        neon = data.stars.element_mass_fractions.neon
    if yvar == "Eu_Fe":
        europium = data.stars.element_mass_fractions.europium
    if yvar == "Ba_Fe":
        barium = data.stars.element_mass_fractions.barium
    if yvar == "Sr_Fe":
        strontium = data.stars.element_mass_fractions.strontium

    if xvar == "O_H":
        O_H = np.log10(oxygen / hydrogen) - O_H_Sun
        O_H[oxygen == 0] = -4  # set lower limit
        O_H[O_H < -4] = -4  # set lower limit
        xval = O_H
    elif xvar == "Fe_H":
        Fe_H = np.log10(iron / hydrogen) - Fe_H_Sun
        Fe_H[iron == 0] = -4  # set lower limit
        Fe_H[Fe_H < -4] = -4  # set lower limit
        xval = Fe_H
    else:
        raise AttributeError(f"Unknown x variable: {xvar}!")

    if yvar == "C_Fe":
        C_Fe = np.log10(carbon / iron) - C_Fe_Sun
        C_Fe[iron == 0] = -2  # set lower limit
        C_Fe[carbon == 0] = -2  # set lower limit
        C_Fe[C_Fe < -2] = -2  # set lower limit
        yval = C_Fe
    elif yvar == "N_Fe":
        N_Fe = np.log10(nitrogen / iron) - N_Fe_Sun
        N_Fe[iron == 0] = -2  # set lower limit
        N_Fe[nitrogen == 0] = -2  # set lower limit
        N_Fe[N_Fe < -2] = -2  # set lower limit
        yval = N_Fe
    elif yvar == "N_O":
        N_O = np.log10(nitrogen / oxygen) - N_O_Sun
        N_O[oxygen == 0] = -2  # set lower limit
        N_O[nitrogen == 0] = -2  # set lower limit
        N_O[N_O < -2] = -2  # set lower limit
        yval = N_O
    elif yvar == "C_O":
        C_O = np.log10(carbon / oxygen) - C_O_Sun
        C_O[oxygen == 0] = -2  # set lower limit
        C_O[carbon == 0] = -2  # set lower limit
        C_O[C_O < -2] = -2  # set lower limit
        yval = C_O
    elif yvar == "O_Fe":
        O_Fe = np.log10(oxygen / iron) - O_Fe_Sun
        O_Fe[iron == 0] = -2  # set lower limit
        O_Fe[oxygen == 0] = -2  # set lower limit
        O_Fe[O_Fe < -2] = -2  # set lower limit
        yval = O_Fe
    elif yvar == "Mg_Fe":
        Mg_Fe = np.log10(magnesium / iron) - Mg_Fe_Sun
        Mg_Fe[iron == 0] = -2  # set lower limit
        Mg_Fe[magnesium == 0] = -2  # set lower limit
        Mg_Fe[Mg_Fe < -2] = -2  # set lower limit
        yval = Mg_Fe
    elif yvar == "Si_Fe":
        Si_Fe = np.log10(silicon / iron) - Si_Fe_Sun
        Si_Fe[iron == 0] = -2  # set lower limit
        Si_Fe[silicon == 0] = -2  # set lower limit
        Si_Fe[Si_Fe < -2] = -2  # set lower limit
        yval = Si_Fe
    elif yvar == "Ne_Fe":
        Ne_Fe = np.log10(neon / iron) - Ne_Fe_Sun
        Ne_Fe[iron == 0] = -2  # set lower limit
        Ne_Fe[neon == 0] = -2  # set lower limit
        Ne_Fe[Ne_Fe < -2] = -2  # set lower limit
        yval = Ne_Fe
    elif yvar == "Eu_Fe":
        Eu_Fe = np.log10(europium / iron) - Eu_Fe_Sun
        Eu_Fe[iron == 0] = -2  # set lower limit
        Eu_Fe[europium == 0] = -2  # set lower limit
        Eu_Fe[Eu_Fe < -2] = -2  # set lower limit
        yval = Eu_Fe
    elif yvar == "Ba_Fe":
        Ba_Fe = np.log10(barium / iron) - Ba_Fe_Sun
        Ba_Fe[iron == 0] = -2  # set lower limit
        Ba_Fe[barium == 0] = -2  # set lower limit
        Ba_Fe[Ba_Fe < -2] = -2  # set lower limit
        yval = Ba_Fe
    elif yvar == "Sr_Fe":
        Sr_Fe = np.log10(strontium / iron) - Sr_Fe_Sun
        Sr_Fe[iron == 0] = -2  # set lower limit
        Sr_Fe[strontium == 0] = -2  # set lower limit
        Sr_Fe[Sr_Fe < -2] = -2  # set lower limit
        yval = Sr_Fe
    elif yvar == "Fe_SNIa_fraction":
        Fe_snia_fraction = unyt_array(np.zeros_like(iron), "dimensionless")
        mask = iron > 0.0
        Fe_snia_fraction[mask] = iron_snia[mask] / iron[mask]
        yval = Fe_snia_fraction
    else:
        raise AttributeError(f"Unknown y variable: {yvar}!")

    return xval, yval


arguments = ScriptArgumentParser(
    description="Creates an [Fe/H] - [C/Fe] plot for stars.",
    additional_arguments={"xvar": "Fe_H", "yvar": "C_Fe", "dataset": None},
)

snapshot_filenames = [
    f"{directory}/{snapshot}"
    for directory, snapshot in zip(arguments.directory_list, arguments.snapshot_list)
]

names = arguments.name_list
output_path = arguments.output_directory
xvar = arguments.xvar
yvar = arguments.yvar
dataset = arguments.dataset

plt.style.use(arguments.stylesheet_location)

simulation_lines = []
simulation_labels = []

fig, ax = plt.subplots()
ax.grid(True)

for isnap, (snapshot_filename, name) in enumerate(zip(snapshot_filenames, names)):

    data = load(snapshot_filename)
    redshift = data.metadata.z

    xval, yval = read_data(data, xvar, yvar)

    # Bins along the X axis to plot the median line
    # The same bins are used for xvar=="Fe_H" and xvar=="O_H"
    bins = np.arange(-4.1, 1, 0.2)
    Min_N_points_per_bin = 11

    xm = 0.5 * (bins[1:] + bins[:-1])
    ym, _, _ = stats.binned_statistic(xval, yval, statistic="median", bins=bins)
    ym1, _, _ = stats.binned_statistic(
        xval, yval, statistic=lambda x: np.percentile(x, 16.0), bins=bins
    )
    ym2, _, _ = stats.binned_statistic(
        xval, yval, statistic=lambda x: np.percentile(x, 84.0), bins=bins
    )
    counts, _, _ = stats.binned_statistic(xval, yval, statistic="count", bins=bins)
    mask = counts >= Min_N_points_per_bin
    xm = xm[mask]
    ym = ym[mask]
    ym1 = ym1[mask]
    ym2 = ym2[mask]

    colour = f"C{isnap}"
    fill_element = ax.fill_between(xm, ym1, ym2, color=colour, alpha=0.2)

    # high zorder, as we want the simulation lines to be on top of everything else
    # we steal the color of the dots to make sure the line has the same color
    simulation_lines.append(
        ax.plot(
            xm,
            ym,
            lw=2,
            color=colour,
            zorder=1000,
            path_effects=[pe.Stroke(linewidth=4, foreground="white"), pe.Normal()],
        )[0]
    )
    simulation_labels.append(f"{name} ($z={redshift:.1f}$)")

path_to_obs_data = f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"
if dataset == "APOGEE":
    if xvar == "Fe_H":
        if yvar == "C_Fe":
            observational_data = (
                f"{path_to_obs_data}/data/StellarAbundances/APOGEE_data_C.hdf5"
            )
        elif yvar == "N_Fe":
            observational_data = (
                f"{path_to_obs_data}/data/StellarAbundances/APOGEE_data_N.hdf5"
            )
        elif yvar == "N_O":
            observational_data = (
                f"{path_to_obs_data}/data/StellarAbundances/APOGEE_data_NO.hdf5"
            )
        elif yvar == "C_O":
            observational_data = (
                f"{path_to_obs_data}/data/StellarAbundances/APOGEE_data_CO.hdf5"
            )
        elif yvar == "O_Fe":
            observational_data = (
                f"{path_to_obs_data}/data/StellarAbundances/APOGEE_data_O.hdf5"
            )
        elif yvar == "Mg_Fe":
            observational_data = (
                f"{path_to_obs_data}/data/StellarAbundances/APOGEE_data_MG.hdf5"
            )
        else:
            raise AttributeError(f"No APOGEE dataset for y variable {yvar}!")
        xmin = -3
        xmax = 1
        ymin = -1
        ymax = 1
    elif xvar == "O_H":
        if yvar == "O_Fe":
            observational_data = (
                f"{path_to_obs_data}/data/StellarAbundances/APOGEE_data_OH.hdf5"
            )
        elif yvar == "N_O":
            observational_data = (
                f"{path_to_obs_data}/data/StellarAbundances/APOGEE_data_NOOH.hdf5"
            )
        elif yvar == "C_O":
            observational_data = (
                f"{path_to_obs_data}/data/StellarAbundances/APOGEE_data_COOH.hdf5"
            )
        elif yvar == "Mg_Fe":
            observational_data = (
                f"{path_to_obs_data}/data/StellarAbundances/APOGEE_data_OHMGFE.hdf5"
            )
        else:
            raise AttributeError(f"No APOGEE dataset for y variable {yvar}!")
        xmin = -3
        xmax = 2
        ymin = -1
        ymax = 2
    else:
        raise AttributeError(f"No APOGEE dataset for x variable {xvar}!")

    # Reading APOGEE data
    with h5py.File(observational_data, "r") as obs_data:
        x = obs_data["x"][:]
        y = obs_data["y"][:]
        GalR = obs_data["GalR"][:]  # in kpc units
        Galz = obs_data["Galz"][:]  # in kpc units

    ngridx = 100
    ngridy = 50

    # Create grid values first.
    xi = np.linspace(xmin, xmax, ngridx)
    yi = np.linspace(ymin, ymax, ngridy)

    # We apply radial & azimuthal cuts, and combine the stellar distributions
    # to give less weight to stars in the solar vicinity. We create a histogram
    # for each distance cut and then combine them
    h, xedges, yedges = make_stellar_abundance_distribution(x, y, GalR, Galz, xi, yi)

    xbins = 0.5 * (xedges[1:] + xedges[:-1])
    ybins = 0.5 * (yedges[1:] + yedges[:-1])

    z = h.T

    binsize = 0.2
    grid_min = np.log10(1) # Note that the histograms have been normalized. Therefore this **does not** indicate a minimum of 1 star per bin!
    grid_max = np.log10(np.ceil(h.max()))
    levels = np.arange(grid_min, grid_max, binsize)
    levels = 10 ** levels

    contour = plt.contour(
        xbins, ybins, z, levels=levels, linewidths=0.5, cmap="winter", zorder=100
    )

    ax.annotate("APOGEE data", (-3.8, -1.3))


    if (xvar == "Fe_H") & (yvar == "O_Fe"):
            observational_data = [f"{path_to_obs_data}/data/StellarAbundances/Cayrel_2004_OFe_FeH.hdf5",
                                  f"{path_to_obs_data}/data/StellarAbundances/Israelian_2004_OFe_FeH.hdf5"]

    elif (xvar == "Fe_H") & (yvar == "C_Fe"):
            observational_data = [f"{path_to_obs_data}/data/StellarAbundances/Cayrel_2004_CFe_FeH.hdf5"]

    elif (xvar == "Fe_H") & (yvar == "N_Fe"):
            observational_data = [f"{path_to_obs_data}/data/StellarAbundances/Cayrel_2004_NFe_FeH.hdf5",
                                  f"{path_to_obs_data}/data/StellarAbundances/Israelian_2004_NFe_FeH.hdf5"]

    elif (xvar == "Fe_H") & (yvar == "N_O"):
            observational_data = [f"{path_to_obs_data}/data/StellarAbundances/Israelian_2004_NO_FeH.hdf5"]

    else:
            observational_data = None

    if not observational_data is None:
        for obs in load_observations(observational_data):
            obs.plot_on_axes(ax)
        observation_legend = ax.legend(markerfirst=True, loc="lower left")
        ax.add_artist(observation_legend)


elif dataset == "GALAH":
    observational_data = (
        f"{path_to_obs_data}/data/StellarAbundances/raw/Buder21_data.hdf5"
    )

    GALAH_data = h5py.File(observational_data, "r")
    galah_edges = np.array(GALAH_data["abundance_bin_edges"])
    if yvar == "C_Fe":
        obs_plane = np.array(GALAH_data["C_enrichment_vs_Fe_abundance"]).T
    elif yvar == "O_Fe":
        obs_plane = np.array(GALAH_data["O_enrichment_vs_Fe_abundance"]).T
    elif yvar == "Mg_Fe":
        obs_plane = np.array(GALAH_data["Mg_enrichment_vs_Fe_abundance"]).T
    elif yvar == "Si_Fe":
        obs_plane = np.array(GALAH_data["Si_enrichment_vs_Fe_abundance"]).T
    elif yvar == "Ba_Fe":
        obs_plane = np.array(GALAH_data["Ba_enrichment_vs_Fe_abundance"]).T
    elif yvar == "Eu_Fe":
        obs_plane = np.array(GALAH_data["Eu_enrichment_vs_Fe_abundance"]).T
    obs_plane[obs_plane < 10] = None

    contour = plt.contour(
        np.log10(obs_plane),
        origin="lower",
        extent=[galah_edges[0], galah_edges[-1], galah_edges[0], galah_edges[-1]],
        zorder=100,
        cmap="winter",
        linewidths=0.5,
    )
    ax.annotate("GALAH data", (-3.8, -1.3))
elif dataset is None:
    if xvar == "Fe_H" and yvar == "O_Fe":
        observational_data = [
            f"{path_to_obs_data}/data/StellarAbundances/Israelian98_data.hdf5",
            f"{path_to_obs_data}/data/StellarAbundances/Mishenina00_data.hdf5",
            f"{path_to_obs_data}/data/StellarAbundances/Bai04_data.hdf5",
            f"{path_to_obs_data}/data/StellarAbundances/Cayrel04_data.hdf5",
            f"{path_to_obs_data}/data/StellarAbundances/Geisler05_data.hdf5",
            f"{path_to_obs_data}/data/StellarAbundances/ZhangZao05_data.hdf5",
            f"{path_to_obs_data}/data/StellarAbundances/Letarte07_data.hdf5",
            f"{path_to_obs_data}/data/StellarAbundances/Sbordone07_data.hdf5",
            f"{path_to_obs_data}/data/StellarAbundances/Koch08_data.hdf5",
        ]
    elif xvar == "Fe_H" and yvar == "Mg_Fe":
        observational_data = glob.glob(
            f"{path_to_obs_data}/data/StellarAbundances/Tolstoy*.hdf5"
        )
    else:
        observational_data = None

    if not observational_data is None:
        for obs in load_observations(observational_data):
            obs.plot_on_axes(ax)
        observation_legend = ax.legend(markerfirst=True, loc="lower left")
        ax.add_artist(observation_legend)
else:
    raise AttributeError(f"Unknown dataset: {dataset}!")

xlabels = {"Fe_H": "[Fe/H]", "O_H": "[O/H]"}
ylabels = {
    "C_Fe": "[C/Fe]",
    "N_Fe": "[N/Fe]",
    "N_O": "[N/O]",
    "C_O": "[C/O]",
    "O_Fe": "[O/Fe]",
    "Mg_Fe": "[Mg/Fe]",
    "Si_Fe": "[Si/Fe]",
    "Ne_Fe": "[Ne/Fe]",
    "Sr_Fe": "[Sr/Fe]",
    "Ba_Fe": "[Ba/Fe]",
    "Eu_Fe": "[Eu/Fe]",
    "Fe_SNIa_fraction": "Fe (SNIa) / Fe (Total)",
}
ax.set_xlabel(xlabels[xvar])
ax.set_ylabel(ylabels[yvar])

ax.set_xlim(-4.0, 2.0)
ylims = {
    "C_Fe": (-1.5, 1.5),
    "N_Fe": (-1.5, 1.5),
    "N_O": (-1.5, 1.5),
    "C_O": (-1.5, 1.5),
    "O_Fe": (-1.5, 1.5),
    "Mg_Fe": (-1.5, 2.0),
    "Si_Fe": (-1.5, 2.0),
    "Ba_Fe": (-1.5, 2.0),
    "Sr_Fe": (-1.5, 2.0),
    "Eu_Fe": (-1.5, 2.0),
    "Ne_Fe": (-1.5, 2.0),
    "Fe_SNIa_fraction": (3.0e-3, 3.0),
}
ax.set_ylim(*ylims[yvar])
if yvar == "Fe_SNIa_fraction":
    ax.set_yscale("log")

simulation_legend = ax.legend(
    simulation_lines, simulation_labels, markerfirst=False, loc="upper right"
)

ax.add_artist(simulation_legend)

output_file = f"{output_path}/stellar_abundances_"
output_file += xvar.replace("_", "")
output_file += "_"
if yvar == "Fe_SNIa_fraction":
    output_file += "Fe_snia_fraction"
else:
    output_file += yvar.replace("_", "")
if dataset == "APOGEE" or dataset == "GALAH":
    output_file += f"_{dataset}"
output_file += ".png"
plt.savefig(output_file)
