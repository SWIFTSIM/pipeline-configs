"""
Plot the stellar mass-metallicity relation
"""

import matplotlib.pyplot as plt
import numpy as np
import unyt

from velociraptor.swift.swift import to_swiftsimio_dataset
from velociraptor.particles import load_groups
from velociraptor import load
from swiftsimio import load as snap_load
from velociraptor.observations import load_observation

import sys

from scipy import stats
from swiftpipeline.argumentparser import ScriptArgumentParser

arguments = ScriptArgumentParser(
    description="Creates stellar mass-[Fe/H] relations using the mean of the log."
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

#Also read some obs data
observational_data_file = f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}/data/GalaxyStellarMassStellarMetallicity/Kirby2013_Data.hdf5"

obs = load_observation(observational_data_file)

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
halo_ids = np.where(stellar_mass >= 1e6)[0]
stellar_mass = stellar_mass[halo_ids]
mean_FeH = []

for halo_id in halo_ids:
    particles, unbound_particles = groups.extract_halo(halo_id, filenames=filenames)
    #data, mask = to_swiftsimio_dataset(particles, snapshot_filename[0], generate_extra_mask=True)
    stars_in_halo = particles.particle_ids[particles.particle_types==4]
    
    # This reads particles using the cell metadata that are around our halo
    data = snap_load(snapshot_filename[0])
    stars_in_snap = data.stars.particle_ids
    Fe = data.stars.element_mass_fractions.iron #this should work but it doesn't
    H = data.stars.element_mass_fractions.hydrogen
    
    # Select only relevant parts
    _, indices_v, indices_p = np.intersect1d(stars_in_halo,stars_in_snap,assume_unique=True, return_indices=True,)
                                            
    Fe = Fe[indices_p]
    H = H[indices_p]
    FeH = Fe[Fe>0]/H[Fe>0] #getting rid of zero metal particles
    mean_FeH = np.append(mean_FeH,np.mean(np.log10(FeH)))

mean_FeH -= Fe_H_Sun


# Begin plotting

fig, ax = plt.subplots()
ax.grid('True')
ax.set_xscale('log')

ax.plot(stellar_mass,mean_FeH,'o',ms=2,label='COLIBRE')

bins = np.arange(6,10,0.2)
bins = 10**bins
ymedians = stats.binned_statistic(stellar_mass, mean_FeH, 'median', bins=bins)
ax.plot(xmedians.statistic,ymedians.statistic,'-',lw=1,color='blue')

ax.errorbar(obs.x,np.log10(obs.y),xerr=obs.x_scatter,marker='o',ms=5, ls='none',color='tab:orange',label='Kirby et al. (2019)')

ax.set_xlim(1e3,1e11)
ax.set_ylim(-4,0)
ax.legend(loc="upper left")
ax.set_xlabel("Stellar Mass [$M_{\odot}$]")
ax.set_ylabel("Mean($\log_{10}$ (Fe/H))-[Fe/H]$_{\odot}$")
fig.savefig(f"{output_path}/stellar_mass_mean_of_log_stellar_FeH.png")

