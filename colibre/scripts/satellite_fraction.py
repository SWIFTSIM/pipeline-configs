import matplotlib.pyplot as plt
import numpy as np
import unyt

from swiftpipeline.argumentparser import ScriptArgumentParser
from velociraptor import load
from velociraptor.observations import load_observations

# Set variables for the figure
mass_bounds = [1e8, 1e12]  # Msun
value_bounds = [0, 1]  # dimensionless
n_mass_bins = 20
min_points_bin = 5

# sSFR below which the galaxy is considered passive
marginal_ssfr = unyt.unyt_quantity(1e-11, units=1 / unyt.year)

# Aperture to use
aperture_size = 50

arguments = ScriptArgumentParser(description="Satellite fraction")

catalogue_filenames = [
    f"{directory}/{catalogue}"
    for directory, catalogue in zip(arguments.directory_list, arguments.catalogue_list)
]

plt.style.use(arguments.stylesheet_location)

# Create a plot using all galaxies, and one using only passive galaxies
for passive_only in [True, False]:

    # Creating plot
    fig, ax = plt.subplots()
    ax.set_xlabel(f"ExclusiveSphere {aperture_size}kpc StellarMass $\\rm [M_\\odot]$")
    ax.set_ylabel("Satellite fraction")
    ax.set_xscale("log")

    # Create mass bins
    mass_bins = np.logspace(*np.log10(mass_bounds), n_mass_bins + 1)
    centers = (mass_bins[1:] + mass_bins[:-1]) / 2

    # Loop over simulations
    for filename, name in zip(catalogue_filenames, arguments.name_list):
        catalogue = load(filename)
        z = catalogue.units.redshift

        # Load quantities, ignoring objects below lower mass limit
        stellar_mass = catalogue.get_quantity(
            f"apertures.mass_star_{aperture_size}_kpc"
        )
        mask = stellar_mass.to("Msun").value > mass_bounds[0]
        stellar_mass = stellar_mass[mask]
        is_central = (catalogue.get_quantity("inputhalos.iscentral") == 1)[mask]

        # Determine which galaxies are passive
        sfr = catalogue.get_quantity(f"apertures.sfr_gas_{aperture_size}_kpc").to(
            "Msun/yr"
        )[mask]
        ssfr = sfr / stellar_mass
        is_passive = ssfr < marginal_ssfr

        # Calculate satellite fraction
        values = -1 * np.ones(n_mass_bins)
        for i_bin in range(n_mass_bins):
            mask = (mass_bins[i_bin] < stellar_mass) & (
                stellar_mass <= mass_bins[i_bin + 1]
            )
            if passive_only:
                mask &= is_passive

            if np.sum(mask) >= min_points_bin:
                values[i_bin] = 1 - (np.sum(is_central[mask]) / np.sum(mask))

        ax.plot(
            centers[values != -1],
            values[values != -1],
            "-",
            label=f"{name} ($z={z:.1f})$",
        )

    ax.legend()
    ax.set_xlim(*mass_bounds)
    ax.set_ylim(*value_bounds)

    fig_name = "satellite_fraction"
    if passive_only:
        fig_name += "_passive"
    fig.savefig(f"{arguments.output_directory}/{fig_name}.png")
