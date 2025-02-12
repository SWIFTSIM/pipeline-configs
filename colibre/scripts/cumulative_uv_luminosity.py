"""
Plots the cumulative UV luminosity.
"""

import matplotlib.pyplot as plt
import numpy as np
import unyt

from velociraptor import load
from swiftpipeline.argumentparser import ScriptArgumentParser

arguments = ScriptArgumentParser(
    description="Creates a plot of normalised cumulative FUV luminosity."
)


# Read in SOAP catalogue
catalogue_filenames = [
    f"{directory}/{catalogue}"
    for directory, catalogue in zip(arguments.directory_list, arguments.catalogue_list)
]

# catalogue_filenames = [ "/users/ariadurr/WORKING_DIR/SWIFT/COLIBRE_RUNS/vIMF/L0025N0188-Density-fE-2p00/SOAP/halo_properties_vimf_0021.hdf5"]

names = arguments.name_list
# names=["L0025N0188"]
output_path = arguments.output_directory

plt.style.use(arguments.stylesheet_location)


data = [load(catalogue_filename) for catalogue_filename in catalogue_filenames]


# Set data features
aperture_size = 50
band= 'FUV'

# Creating plot
fig, ax = plt.subplots()
ax.set(
    xlabel=f"{band}-band AB magnitudes ({aperture_size} kpc) ",
    ylabel="Cumulative Luminosity Fraction",
    xlim=[-9.0,-25.0],
    ylim=[-0.05,1.05],
    # yscale="log"
)

# Generate cumulative luminosity value
for color, (catalogue, name) in enumerate(zip(data, names)):

    luminosity = catalogue.get_quantity(
        f"corrected_stellar_luminosities.{band}_luminosity_{aperture_size}_kpc").to_value()
    
    luminosity = np.sort(luminosity)[::-1]
    luminosity_cumulative = np.cumsum( luminosity )
    luminosity_cumulative /= np.max(luminosity_cumulative)

    magnitude = np.array( [ -2.5 * np.log10(lum) if lum > 0 else 0 for lum in luminosity ] )

    ax.plot(
        magnitude,
        luminosity_cumulative,
        label=name,
        color=f"C{color}"
    )

    # Add 0.5L marker
    M_UV_half = magnitude[np.where(luminosity_cumulative < 0.5)[0][-1]]
    
    ax.vlines(
        M_UV_half,
        ymin=-0.5,
        ymax=0.5,
        linestyle='dashed',
        color=f"C{color}",
        alpha=0.5
    )

    ax.annotate(
        "$M_{\\rm UV}$ = %.1f" % (M_UV_half),
        xy=(0.15,0.75-color/20),
        xycoords='figure fraction',
        color=f"C{color}",
        fontsize=8,
        alpha=0.7
    )

ax.annotate(
    "0.5 $L_{\\rm total}$ :",
    xy=(0.15,0.8),
    xycoords='figure fraction',
    color="black",
    fontsize=8,
    alpha=0.7
)

ax.legend(loc='lower left')

fig_name = f"uv_corrected_luminosity_cumulative_sum"

fig.savefig(f"{arguments.output_directory}/{fig_name}.png")

