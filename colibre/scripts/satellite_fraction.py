import matplotlib.pyplot as plt
import numpy as np
import unyt

from swiftpipeline.argumentparser import ScriptArgumentParser
from velociraptor import load
from velociraptor.observations import load_observations

# Set variables for the figure
mass_bounds = [1e8, 1e12]  # Msun
value_bounds = [0, 0.5]    # dimensionless
n_mass_bins = 20
min_points_bin = 5

arguments = ScriptArgumentParser(
    description="Satellite fraction"
)

catalogue_filenames = [
    f"{directory}/{catalogue}"
    for directory, catalogue in zip(arguments.directory_list, arguments.catalogue_list)
]

plt.style.use(arguments.stylesheet_location)

aperture_size = 50

# Create a plot using all galaxies, and one using only passive galaxies
for passive_only in [True, False]:

    # Creating plot
    fig, ax = plt.subplots()
    ax.set_xlabel(f"ExclusiveSphere {aperture_size}kpc StellarMass $\\rm [M_\odot]$")
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
        star_mass = catalogue.get_quantity(f"apertures.mass_star_{aperture_size}_kpc")
        mask = star_mass > mass_bounds[0] * unyt.Msun
        star_mass = star_mass[mask]
        sfr = catalogue.get_quantity(f"apertures.sfr_gas_{aperture_size}_kpc").to(
            "Msun/yr"
        )[mask]
        is_central = (catalogue.get_quantity("inputhalos.iscentral") == 1)[mask]
        is_passive = (catalogue.get_quantity(f"derived_quantities.is_passive_{aperture_size}_kpc") == 1)[mask]

        # Calculate satellite fraction
        values = -1 * np.ones(n_mass_bins)
        for i_bin in range(n_mass_bins):
            mask = (mass_bins[i_bin] < star_mass) & (star_mass <= mass_bins[i_bin + 1])
            if passive_only:
                mask &= is_passive

            if np.sum(mask) >= min_points_bin:
                values[i_bin] = (1 - np.sum(is_central[mask])) / np.sum(mask)

        ax.plot(
            centers[values != -1], values[values != -1], "-", label=f"{name} ($z={z:.1f})$"
        )

    ax.legend()
    ax.set_xlim(*mass_bounds)
    ax.set_ylim(*value_bounds)

    fig_name = "satellite_fraction"
    if passive_only:
        fig_name += "_passive"
    fig.savefig(
        f"{arguments.output_directory}/{fig_name}.png"
    )
