import matplotlib.pyplot as plt
import numpy as np
import unyt

from swiftpipeline.argumentparser import ScriptArgumentParser
from velociraptor import load

# Set variables for the figure
mass_bounds = [1e9, 1e12]  # Msun
value_bounds = [-0.5, 1.0]  # dimensionless
n_mass_bins = 10
min_points_bin = 5

arguments = ScriptArgumentParser(
    description="Metallicity difference - Passive vs Active"
)

catalogue_filenames = [
    f"{directory}/{catalogue}"
    for directory, catalogue in zip(arguments.directory_list, arguments.catalogue_list)
]

plt.style.use(arguments.stylesheet_location)

aperture_size = 50
solar_fe_abundance = 2.82e-5

# Creating plot
fig, ax = plt.subplots()
ax.set_xlabel(f"ExclusiveSphere {aperture_size}kpc StellarMass $[M_\odot]$")
ax.set_ylabel(
    "Stellar [Fe/H]$_{\mathrm{passive}}$ - Stellar [Fe/H]$_{\mathrm{active}}$"
)
ax.set_xscale("log")

z_to_plot = set()
obs_data_z = np.array([0, 0.5, 1, 2])
for filename, name in zip(catalogue_filenames, arguments.name_list):
    catalogue = load(filename)

    # Determining which redshift observational data to plot
    z = catalogue.units.redshift
    if z <= 3:
        z_to_plot.add(obs_data_z[np.argmin(np.abs(obs_data_z - z))])

    # Load quantities, ignoring objects below mass bin
    star_mass = catalogue.get_quantity(f"apertures.mass_star_{aperture_size}_kpc")
    mask = star_mass > mass_bounds[0] * star_mass.units
    star_mass = star_mass[mask]
    lin_Fe_over_H_times_star_mass = catalogue.get_quantity(
        f"lin_element_ratios_times_masses.lin_Fe_over_H_times_star_mass_{aperture_size}_kpc"
    )[mask]
    sfr = catalogue.get_quantity(f"apertures.sfr_gas_{aperture_size}_kpc").to(
        "Msun/yr"
    )[mask]
    is_central = (catalogue.get_quantity("structure_type.structuretype") == 10)[mask]

    # Compute stellar-mass weighted Fe over H
    Fe_over_H = lin_Fe_over_H_times_star_mass / star_mass

    # Convert to units used in observations
    Fe_abundance = np.log10(Fe_over_H / solar_fe_abundance)

    # Calculate if galaxy is passive or star-forming using definition from Lyu+23
    log_sfr_ms = 0.8 * np.log10(star_mass) - 8.1
    # Galaxies with zero sfr will be defined as passive
    delta_ms = -2 * np.ones(star_mass.shape[0])
    delta_ms[sfr != 0] = np.log10(sfr[sfr != 0]) - log_sfr_ms[sfr != 0]

    # Calculate difference in each mass bin
    mass_bins = np.logspace(*np.log10(mass_bounds), n_mass_bins + 1)
    centers = (mass_bins[1:] + mass_bins[:-1]) / 2
    values = np.zeros(n_mass_bins)
    for i_bin in range(n_mass_bins):
        mask = (mass_bins[i_bin] < star_mass) & (star_mass <= mass_bins[i_bin + 1])
        mask &= is_central
        passive_abundance = Fe_abundance[mask & (delta_ms < -1.5)]
        star_forming_abundance = Fe_abundance[mask & (delta_ms > -0.5)]
        if (passive_abundance.shape[0] >= min_points_bin) and (
            star_forming_abundance.shape[0] >= min_points_bin
        ):
            values[i_bin] = np.median(passive_abundance) - np.median(
                star_forming_abundance
            )
    ax.plot(
        centers[values != 0], values[values != 0], "-", label=f"{name} ($z={z:.1f})$"
    )

# Loading and plotting observational data
obs_data_dir = f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}/data/GalaxyStellarMassStellarMetallicity/raw/"
for z in sorted(z_to_plot):
    for filename, label in [
        ("Peng15.txt", "Peng et al. (2015)"),
        ("Trussler20.txt", "Trussler et al. (2020)"),
        ("Lyu23.txt", "Lyu et al. (2023)"),
    ]:
        obs_data = np.loadtxt(f"{obs_data_dir}{filename}")
        mask = obs_data[:, 0] == z
        if np.sum(mask) == 0:
            continue
        ax.errorbar(
            10 ** obs_data[:, 1][mask],
            obs_data[:, 2][mask],
            yerr=obs_data[:, 3][mask],
            label=f"{label} ($z={z:.1f})$",
        )

ax.legend()
ax.set_xlim(*mass_bounds)
ax.set_ylim(*value_bounds)

fig.savefig(f"{arguments.output_directory}/passive_active_metallicity_difference.png")
