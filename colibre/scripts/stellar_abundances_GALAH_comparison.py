"""
Plots the stellar abundances ([Fe/H] vs [X/Fe]) for a given snapshot
"""
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import unyt
import glob
from swiftsimio import load
from swiftpipeline.argumentparser import ScriptArgumentParser
from velociraptor.observations import load_observations
import h5py as h5

def read_data(data, enrichment_element):
    """
    Grabs the data
    """

    A = {'H': unyt.mh,
         'C': 12.0107 * unyt.mp,
         'O': 15.9994 * unyt.mp,
         'Mg': 24.305 * unyt.mp,
         'Si': 28.0855 * unyt.mp,
         'Fe': 55.845 * unyt.mp,
         'Eu': 151.964 * unyt.mp}

    sun_ab = {'C': 8.43,
              'O': 8.69,
              'Mg': 7.6,
              'Si': 7.51,
              'Fe': 7.5,
              'Eu': 0.52}

    sun_en = {'C':  sun_ab['C'] - sun_ab['Fe'] + np.log10(A['C']/A['Fe']),
              'O':  sun_ab['O'] - sun_ab['Fe'] + np.log10(A['O']/A['Fe']),
              'Mg': sun_ab['Mg'] - sun_ab['Fe'] + np.log10(A['Mg']/A['Fe']),
              'Si': sun_ab['Si'] - sun_ab['Fe'] + np.log10(A['Si']/A['Fe']),
              'Eu': sun_ab['Eu'] - sun_ab['Fe'] + np.log10(A['Eu']/A['Fe'])}

    labels = {'C': 'carbon',
              'O': 'oxygen',
              'Mg': 'magnesium',
              'Si': 'silicon',
              'Eu': 'europium'}
    
    element = getattr(data.stars.element_mass_fractions, labels[enrichment_element])
    iron = data.stars.element_mass_fractions.iron
    hydrogen = data.stars.element_mass_fractions.hydrogen

    Fe_H = 12 + np.log10(iron / hydrogen) - sun_ab['Fe'] - np.log10(55.845)
    X_Fe = np.log10(element / iron) - sun_en[enrichment_element]

    Fe_H[iron == 0] = -7  # set lower limit
    Fe_H[Fe_H < -7] = -7  # set lower limit

    X_Fe[iron == 0] = -2  # set lower limit
    X_Fe[element == 0] = -2  # set lower limit
    X_Fe[X_Fe < -2] = -2  # set lower limit

    return Fe_H, X_Fe


arguments = ScriptArgumentParser(
    description="Creates an [Fe/H] - [X/Fe] plot for stars.",
    additional_arguments={
            "enrichment_element": "Mg"
        }
)

enrichment_element = arguments.enrichment_element

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

    Fe_H, X_Fe = read_data(data, enrichment_element)

    # low zorder, as we want these points to be in the background
    dots = ax.plot(Fe_H, X_Fe, ".", markersize=0.2, alpha=0.2, zorder=-99)[0]

    # Bins along the X axis (Fe_H) to plot the median line
    bins = np.arange(-7.2, 1, 0.2)
    ind = np.digitize(Fe_H, bins)

    xm, ym = [], []
    Min_N_points_per_bin = 11

    for i in range(1, len(bins)):
        in_bin_idx = ind == i
        N_data_points_per_bin = np.sum(in_bin_idx)
        if N_data_points_per_bin >= Min_N_points_per_bin:
            xm.append(np.median(Fe_H[in_bin_idx]))
            ym.append(np.median(X_Fe[in_bin_idx]))

    # high zorder, as we want the simulation lines to be on top of everything else
    # we steal the color of the dots to make sure the line has the same color
    simulation_lines.append(
        ax.plot(
            xm,
            ym,
            lw=2,
            color=dots.get_color(),
            zorder=1000,
            path_effects=[pe.Stroke(linewidth=4, foreground="white"), pe.Normal()],
        )[0]
    )
    simulation_labels.append(f"{name} ($z={redshift:.1f}$)")

# We select all Tolstoy* files containing FeH-XFe.
path_to_obs_data = f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"
GALAH_data = h5.File(f"{path_to_obs_data}/data/StellarAbundances/Buder21_data.hdf5", 'r')
obs_plane = np.array(GALAH_data[f"{enrichment_element}_enrichment_vs_Fe_abundance"]).T
obs_plane[obs_plane < 10] = None
edges = np.array(GALAH_data["abundance_bin_edges"])

contour = ax.contour(np.log10(obs_plane), origin='lower',
                     extent=[edges[0], edges[-1], edges[0], edges[-1]])
contour.collections[0].set_label('GALAH DR3')

ax.set_xlabel("[Fe/H]")
ax.set_ylabel(f"[{enrichment_element}/Fe]")

ax.set_ylim(-1.5, 1.5)
ax.set_xlim(-4, 2.0)

observation_legend = ax.legend(markerfirst=True, loc="upper left")

ax.add_artist(observation_legend)

simulation_legend = ax.legend(
    simulation_lines, simulation_labels, markerfirst=False, loc="lower left"
)

ax.add_artist(simulation_legend)

plt.savefig(f"{output_path}/stellar_abundances_GALAH_FeH_{enrichment_element}Fe.png")
