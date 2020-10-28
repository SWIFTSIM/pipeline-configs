"""
Plot the stellar mass-metallicity relation
"""

import matplotlib.pyplot as plt
import numpy as np
import unyt

from velociraptor.swift.swift import to_swiftsimio_dataset
from velociraptor.particles import load_groups
from velociraptor import load

import sys

from swiftpipeline.argumentparser import ScriptArgumentParser

arguments = ScriptArgumentParser(
    description="Creates stellar mass-[Fe/H] relations using either the mean of log or log of mean, pick your favourite."
)

snapshot_filename = [
    f"{directory}/{snapshot}"
    for directory, snapshot in zip(arguments.directory_list, arguments.snapshot_list)
]

velociraptor_filename = [
    f"{directory}/{catalogue}"
    for directory, catalogue in zip(arguments.directory_list, arguments.catalogue_list)
]

names = arguments.name_list
output_path = arguments.output_directory

plt.style.use(arguments.stylesheet_location)

velociraptor_properties = velociraptor_filename[0]
velociraptor_groups = velociraptor_properties.replace("properties", "catalog_groups")

filenames = {
    "parttypes_filename": velociraptor_properties.replace(
        "properties", "catalog_parttypes"),
    "particles_filename": velociraptor_properties.replace(
        "properties", "catalog_particles"),
    "unbound_parttypes_filename": velociraptor_properties.replace(
        "properties", "catalog_parttypes.unbound"),
    "unbound_particles_filename": velociraptor_properties.replace(
        "properties", "catalog_particles.unbound"),
}

#Before reading a bit more I keep some quantities here
mp_in_cgs = 1.6737236e-24
mH_in_cgs = 1.00784 * mp_in_cgs
mFe_in_cgs = 55.845 * mp_in_cgs
Fe_H_Sun = 7.5  # Asplund et al. (2009)
Fe_H_Sun = Fe_H_Sun - 12.0 - np.log10(mH_in_cgs / mFe_in_cgs)

catalogue = load(velociraptor_properties)
groups = load_groups(velociraptor_groups, catalogue)

# Let's stellar mass and impose minimum #
stellar_mass = catalogue.apertures.mass_star_30_kpc.to("Solar_Mass")
halo_ids = np.where(stellar_mass >= 1e7)[0]
stellar_mass = stellar_mass[halo_ids]
FeH_mean = []
mean_FeH = []

for halo_id in halo_ids:
    particles, unbound_particles = groups.extract_halo(halo_id, filenames=filenames)
    
    # This reads particles using the cell metadata that are around our halo
    data, mask = to_swiftsimio_dataset(particles, snapshot_filename[0], generate_extra_mask=True)
    Fe = data.stars.element_mass_fractions.iron #this should work but it doesn't
    H = data.stars.element_mass_fractions.hydrogen

    FeH = Fe[Fe>0]/H[Fe>0] #getting rid of zero metal particles
    FeH_mean = np.append(FeH_mean,np.log10(np.mean(FeH)))
    mean_FeH = np.append(FeH_mean,np.mean(np.log10(FeH)))

FeH_mean -= Fe_H_Sun
mean_FeH -= Fe_H_Sun


# Begin plotting

fig, ax = plt.subplots()

ax.loglog()

ax.plot(stellar_mass,mean_FeH,'o',ms=5)

#ax.legend(loc="upper right")
ax.set_xlabel("Stellar Mass [$M_{\odot}$]")
ax.set_ylabel("$\log_{10}$~Mean(Fe/H)-[Fe/H]$_{\odot}$")
fig.savefig(f"{output_path}/stellar_mass_log_of_mean_stellar_FeH.png")


# Begin also second plot

fig, ax = plt.subplots()

ax.loglog()

ax.plot(stellar_mass,mean_FeH,'o',ms=5)

ax.set_xlabel("Stellar Mass [$M_{\odot}$]")
ax.set_ylabel("Mean($\log_{10}$~(Fe/H))-[Fe/H]$_{\odot}$")
fig.savefig(f"{output_path}/stellar_mass_mean_of_log_stellar_FeH.png")
