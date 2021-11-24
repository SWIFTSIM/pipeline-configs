import h5py


import numpy as np
from swiftsimio import load
from numba import jit
from unyt import g, cm, mp, erg, s

class interpolate:
    def init(self):
        pass

    def load_table(self, obs_data_dir, band):
        self.table = h5py.File(obs_data_dir + 'X_Ray_table_redshifts.hdf5', 'r')
        self.X_Ray = self.table[band]['emissivities'][()]
        self.He_bins = self.table['/Bins/He_bins'][()]
        self.missing_elements = self.table['/Bins/Missing_element'][()]

        self.density_bins = self.table['/Bins/Density_bins/'][()]
        self.temperature_bins = self.table['/Bins/Temperature_bins/'][()]
        self.redshift_bins = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        self.dn = 0.2
        self.dT = 0.1

        self.solar_metallicity = self.table['/Bins/Solar_metallicities/'][()]

@jit(nopython = True)
def find_dx(subdata, bins, idx_0):
    dx_p = np.zeros(len(subdata))
    for i in range(len(subdata)):
        if (subdata[i] < bins[0]):
            dx_p[i] = 0
        elif (subdata[i] > bins[-1]):
            dx_p[i] = np.abs(bins[-1] - bins[-2])
        else:
            dx_p[i] = np.abs(bins[idx_0[i]] - subdata[i])

    return dx_p

@jit(nopython = True)
def find_idx(subdata, bins):
    idx_p = np.zeros((len(subdata), 2))
    for i in range(len(subdata)):
        if (subdata[i] < bins[0]):
            idx_p[i, :] = np.array([0, 1])
        elif (subdata[i] > bins[-1]):
            idx_p[i, :] = np.array([len(bins)-2, len(bins)-1])
        else:
            idx_p[i, :] = np.sort(np.argsort(np.abs(bins - subdata[i]))[:2])

    return idx_p

@jit(nopython = True)
def find_idx_z(subdata, bins):
    idx_p = np.zeros(2)
    if (subdata < bins[0]):
        idx_p = np.array([0, 1])
    elif (subdata > bins[-1]):
        idx_p = np.array([len(bins)-2, len(bins)-1])
    else:
        idx_p = np.sort(np.argsort(np.abs(bins - subdata))[:2])

    return idx_p

@jit(nopython = True)
def find_dx_z(subdata, bins, idx_0):
    dx_p = 0
    if (subdata < bins[0]):
        dx_p = 0
    elif (subdata > bins[-1]):
        dx_p = 1
    else:
        dx_p = np.abs(subdata - bins[idx_0]) / (bins[idx_0 + 1] - bins[idx_0])

    return dx_p

@jit(nopython = True)
def find_idx_he(subdata, bins):
    num_bins = len(bins)
    idx_p = np.zeros((len(subdata), 2))
    for i in range(len(subdata)):
        
        # When closest to the highest bin, or above the highest bin, return the one but highest bin,
        # otherwise we will select a second bin which is outside the binrange
        bin_below = min(np.argsort(np.abs(bins[bins <= subdata[i]] - subdata[i]))[0], num_bins - 2)
        idx_p[i, :] = np.array([bin_below, bin_below + 1])

    return idx_p

@jit(nopython = True)
def find_dx_he(subdata, bins, idx_0):
    dx_p = np.zeros(len(subdata))
    for i in range(len(subdata)):
        if (subdata[i] < bins[0]):
            dx_p[i] = 0
        elif (subdata[i] > bins[-1]):
            dx_p[i] = 1
        else:
            dx_p[i] = np.abs(subdata[i] - bins[idx_0[i]]) / (bins[idx_0[i]+1] - bins[idx_0[i]])
        # dx_p1[i] = np.abs(bins[idx_0[i+1]] - subdata[i])

    return dx_p

@jit(nopython = True)
def get_table_interp(dn, dT, dx_T, dx_n, idx_T, idx_n, idx_he, dx_he, idx_z, dx_z, X_Ray, abundance_to_solar, z_index):
    f_n_T_Z = np.zeros(len(idx_n[:, 0]))

    t_z = (1 - dx_z)
    d_z = dx_z

    for i in range(len(idx_n[:, 0])):
        t_T = (dT - dx_T[i]) / dT
        d_T = dx_T[i] / dT

        t_n = (dn - dx_n[i]) / dn
        d_n = dx_n[i] / dn

        d_he = dx_he[i]
        t_he = (1 - dx_he[i])

        # if i == len(idx_n[:, 0]) - 1:
        #     print(t_T, d_T, t_n, d_n, d_he, t_he)
        # print(X_Ray.shape, z_index, idx_he, idx_T, idx_n)
        f_n_T = t_T * t_n * t_he * t_z * X_Ray[idx_z[0], idx_he[i, 0], :, idx_T[i, 0], idx_n[i, 0]]
        f_n_T += t_T * t_n * d_he * t_z * X_Ray[idx_z[0], idx_he[i, 1], :, idx_T[i, 0], idx_n[i, 0]]
        f_n_T += t_T * d_n * t_he * t_z * X_Ray[idx_z[0], idx_he[i, 0], :, idx_T[i, 0], idx_n[i, 1]]
        f_n_T += d_T * t_n * t_he * t_z * X_Ray[idx_z[0], idx_he[i, 0], :, idx_T[i, 1], idx_n[i, 0]]
        f_n_T += t_T * d_n * d_he * t_z * X_Ray[idx_z[0], idx_he[i, 1], :, idx_T[i, 0], idx_n[i, 1]]
        f_n_T += d_T * t_n * d_he * t_z * X_Ray[idx_z[0], idx_he[i, 1], :, idx_T[i, 1], idx_n[i, 0]]
        f_n_T += d_T * d_n * t_he * t_z * X_Ray[idx_z[0], idx_he[i, 0], :, idx_T[i, 1], idx_n[i, 1]]
        f_n_T += d_T * d_n * d_he * t_z * X_Ray[idx_z[0], idx_he[i, 1], :, idx_T[i, 1], idx_n[i, 1]]

        f_n_T += t_T * t_n * t_he * d_z * X_Ray[idx_z[1], idx_he[i, 0], :, idx_T[i, 0], idx_n[i, 0]]
        f_n_T += t_T * t_n * d_he * d_z * X_Ray[idx_z[1], idx_he[i, 1], :, idx_T[i, 0], idx_n[i, 0]]
        f_n_T += t_T * d_n * t_he * d_z * X_Ray[idx_z[1], idx_he[i, 0], :, idx_T[i, 0], idx_n[i, 1]]
        f_n_T += d_T * t_n * t_he * d_z * X_Ray[idx_z[1], idx_he[i, 0], :, idx_T[i, 1], idx_n[i, 0]]
        f_n_T += t_T * d_n * d_he * d_z * X_Ray[idx_z[1], idx_he[i, 1], :, idx_T[i, 0], idx_n[i, 1]]
        f_n_T += d_T * t_n * d_he * d_z * X_Ray[idx_z[1], idx_he[i, 1], :, idx_T[i, 1], idx_n[i, 0]]
        f_n_T += d_T * d_n * t_he * d_z * X_Ray[idx_z[1], idx_he[i, 0], :, idx_T[i, 1], idx_n[i, 1]]
        f_n_T += d_T * d_n * d_he * d_z * X_Ray[idx_z[1], idx_he[i, 1], :, idx_T[i, 1], idx_n[i, 1]]

        # if i == len(idx_n[:, 0]) - 1:
        #     print(f_n_T[-1])

        #Apply linear scaling for removed metals
        f_n_T_Z_temp = f_n_T[-1]
        for j in range(len(f_n_T) - 1):
            f_n_T_Z_temp -= (f_n_T[-1] - f_n_T[j]) * abundance_to_solar[i, j]

        f_n_T_Z[i] = f_n_T_Z_temp

    return f_n_T_Z

def interpolate_X_Ray(data_n, data_T, element_mass_fractions, redshift, band = 'soft', fill_value = None):

    z_options = np.array([0, 0.12, 0.25, 0.37, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
    z_index = np.argmin(np.abs(z_options - redshift))

    #Initialise interpolation class
    interp = interpolate()
    interp.load_table(band)

    #Initialise the emissivity array which will be returned
    emissivities = np.zeros_like(data_n, dtype = float)

    #Create density mask, round to avoid numerical errors
    density_mask = (data_n >= np.round(interp.density_bins.min(), 1)) & (data_n <= np.round(interp.density_bins.max(), 1))
    #Create temperature mask, round to avoid numerical errors
    temperature_mask = (data_T >= np.round(interp.temperature_bins.min(), 1)) & (data_T <= np.round(interp.temperature_bins.max(), 1))

    #Combine masks
    joint_mask = density_mask & temperature_mask

    #Check if within density and temperature bounds
    density_bounds = np.sum(density_mask) == density_mask.shape[0]
    temperature_bounds = np.sum(temperature_mask) == temperature_mask.shape[0]
    if ~(density_bounds & temperature_bounds):
        #If no fill_value is set, return an error with some explanation
        if fill_value == None:
            print('Temperature or density are outside of the interpolation range and no fill_value is supplied')
            print('Temperature ranges between log(T) = 5 and log(T) = 9.5')
            print('Density ranges between log(nH) = -8 and log(nH) = 6')
            print('Set the kwarg "fill_value = some value" to set all particles outside of the interpolation range to "some value"')
            print('Or limit your particle data set to be within the interpolation range')
            print('\n')
            print('Interpolation will be capped, all particles outside of the interpolation range will be set to the edges of that range')
            joint_mask = data_n > -100 #Everything
        else:
            emissivities[~joint_mask] = fill_value





    

    mass_fraction = np.zeros((len(data_n[joint_mask]), 9))

    #get individual mass fraction
    mass_fraction[:, 0] = element_mass_fractions.hydrogen[joint_mask]
    mass_fraction[:, 1] = element_mass_fractions.helium[joint_mask]
    mass_fraction[:, 2] = element_mass_fractions.carbon[joint_mask]
    mass_fraction[:, 3] = element_mass_fractions.nitrogen[joint_mask]
    mass_fraction[:, 4] = element_mass_fractions.oxygen[joint_mask]
    mass_fraction[:, 5] = element_mass_fractions.neon[joint_mask]
    mass_fraction[:, 6] = element_mass_fractions.magnesium[joint_mask]
    mass_fraction[:, 7] = element_mass_fractions.silicon[joint_mask]
    mass_fraction[:, 8] = element_mass_fractions.iron[joint_mask]


    # # From Cooling tables, integrate into X-Ray tables in a future version
    # max_mass_fractions = np.array([7.5597578e-01, 2.5954419e-01, 7.1018077e-03, 2.0809614e-03,
    #                                    1.7216533e-02, 3.7735226e-03, 2.1262325e-03, 1.9969980e-03,
    #                                     3.8801527e-03])

    # # At the moment the tables only go up to 0.5 * solar metallicity for He
    # # Enforce this for all metals to have consistency 
    # clip_mass_fractions = max_mass_fractions
    # # Reset such to mass fractions of Hydrogen = 1
    # clip_mass_fractions /= clip_mass_fractions[0]
    # mass_fraction = np.clip(mass_fraction, a_min = 0, a_max = clip_mass_fractions)
    

    #Find density offsets
    idx_n = find_idx(data_n[joint_mask], interp.density_bins)
    dx_n = find_dx(data_n[joint_mask], interp.density_bins, idx_n[:, 0].astype(int))

    #Find temperature offsets
    idx_T = find_idx(data_T[joint_mask], interp.temperature_bins)
    dx_T = find_dx(data_T[joint_mask], interp.temperature_bins, idx_T[:, 0].astype(int))

    #Find element offsets
    #mass of ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'iron']
    element_masses = [1, 4.0026, 12.0107, 14.0067, 15.999, 20.1797, 24.305, 28.0855, 55.845]

    #Calculate the abundance wrt to solar
    # abundances = mass_fraction / np.array(element_masses)
    abundances = (mass_fraction / np.expand_dims(mass_fraction[:, 0], axis = 1)) / np.array(element_masses)

    # Clip to solar metallicity
    # abundances = np.clip(abundances, a_min = 0, a_max = 10**interp.solar_metallicity)

    #Calculate abundance offsets using solar mass fractions
    # This should be less susceptible to changes in the hydrogen mass fraction
    solar_mass_fractions = [7.0030355e-01, 2.5954419e-01, 7.1018077e-03, 2.0809614e-03,
                             1.7216533e-02, 3.7735226e-03, 2.1262325e-03, 1.9969980e-03,
                             3.8801527e-03]
    abundance_to_solar = 1 - mass_fraction / solar_mass_fractions

    # Calculate abundance offsets using solar abundances
    # abundance_to_solar = 1 - abundances / 10**interp.solar_metallicity

    abundance_to_solar = np.c_[abundance_to_solar[:, :-1], abundance_to_solar[:, -2], abundance_to_solar[:, -2], abundance_to_solar[:, -1]] #Add columns for Calcium and Sulphur and add Iron at the end

    #Find helium offsets
    idx_he = find_idx_he(np.log10(abundances[:, 1]), interp.He_bins)
    dx_he = find_dx_he(np.log10(abundances[:, 1]), interp.He_bins, idx_he[:, 0].astype(int))


    # Find redshift offsets
    idx_z = find_idx_z(redshift, interp.redshift_bins)
    dx_z = find_dx_z(redshift, interp.redshift_bins, idx_z[0].astype(int))

    # print('Start interpolation')
    # print(interp.X_Ray.shape, z_index, idx_he[0,:], idx_T[0,:], idx_n[0,:])
    emissivities[joint_mask] = get_table_interp(interp.dn, interp.dT, dx_T, dx_n, idx_T.astype(int), idx_n.astype(int), idx_he.astype(int), dx_he, idx_z.astype(int), dx_z, interp.X_Ray, abundance_to_solar[:, 2:], z_index)

    return emissivities



def interpolate_X_Ray_pandas(data_n, data_T, df_elements, redshift, obs_data_dir, band = 'soft', fill_value = None):

    z_options = np.array([0, 0.12, 0.25, 0.37, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
    z_index = np.argmin(np.abs(z_options - redshift))

    #Initialise interpolation class
    interp = interpolate()
    interp.load_table(obs_data_dir, band)

    #Initialise the emissivity array which will be returned
    emissivities = np.zeros_like(data_n, dtype = float)

    #Create density mask, round to avoid numerical errors
    density_mask = (data_n >= np.round(interp.density_bins.min(), 1)) & (data_n <= np.round(interp.density_bins.max(), 1))
    #Create temperature mask, round to avoid numerical errors
    temperature_mask = (data_T >= np.round(interp.temperature_bins.min(), 1)) & (data_T <= np.round(interp.temperature_bins.max(), 1))

    #Combine masks
    joint_mask = density_mask & temperature_mask

    #Check if within density and temperature bounds
    density_bounds = np.sum(density_mask) == density_mask.shape[0]
    temperature_bounds = np.sum(temperature_mask) == temperature_mask.shape[0]
    if ~(density_bounds & temperature_bounds):
        #If no fill_value is set, return an error with some explanation
        if fill_value == None:
            print('Temperature or density are outside of the interpolation range and no fill_value is supplied')
            print('Temperature ranges between log(T) = 5 and log(T) = 9.5')
            print('Density ranges between log(nH) = -8 and log(nH) = 6')
            print('Set the kwarg "fill_value = some value" to set all particles outside of the interpolation range to "some value"')
            print('Or limit your particle data set to be within the interpolation range')
            print('\n')
            print('Interpolation will be capped, all particles outside of the interpolation range will be set to the edges of that range')
            joint_mask = data_n > -100 #Everything
        else:
            emissivities[~joint_mask] = fill_value


    #get individual mass fraction
    mass_fraction = df_elements.to_numpy()

    # # From Cooling tables, integrate into X-Ray tables in a future version
    # max_mass_fractions = np.array([7.5597578e-01, 2.5954419e-01, 7.1018077e-03, 2.0809614e-03,
    #                                    1.7216533e-02, 3.7735226e-03, 2.1262325e-03, 1.9969980e-03,
    #                                     3.8801527e-03])

    # # At the moment the tables only go up to 0.5 * solar metallicity for He
    # # Enforce this for all metals to have consistency 
    # clip_mass_fractions = max_mass_fractions
    # # Reset such to mass fractions of Hydrogen = 1
    # clip_mass_fractions /= clip_mass_fractions[0]
    # mass_fraction = np.clip(mass_fraction, a_min = 0, a_max = clip_mass_fractions)
    

    #Find density offsets
    idx_n = find_idx(data_n[joint_mask], interp.density_bins)
    dx_n = find_dx(data_n[joint_mask], interp.density_bins, idx_n[:, 0].astype(int))

    #Find temperature offsets
    idx_T = find_idx(data_T[joint_mask], interp.temperature_bins)
    dx_T = find_dx(data_T[joint_mask], interp.temperature_bins, idx_T[:, 0].astype(int))

    #Find element offsets
    #mass of ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'iron']
    element_masses = [1, 4.0026, 12.0107, 14.0067, 15.999, 20.1797, 24.305, 28.0855, 55.845]

    #Calculate the abundance wrt to solar
    # abundances = mass_fraction / np.array(element_masses)
    abundances = (mass_fraction / np.expand_dims(mass_fraction[:, 0], axis = 1)) / np.array(element_masses)

    # Clip to solar metallicity
    # abundances = np.clip(abundances, a_min = 0, a_max = 10**interp.solar_metallicity)

    #Calculate abundance offsets using solar mass fractions
    # This should be less susceptible to changes in the hydrogen mass fraction
    solar_mass_fractions = [7.0030355e-01, 2.5954419e-01, 7.1018077e-03, 2.0809614e-03,
                             1.7216533e-02, 3.7735226e-03, 2.1262325e-03, 1.9969980e-03,
                             3.8801527e-03]
    abundance_to_solar = 1 - mass_fraction / solar_mass_fractions

    # Calculate abundance offsets using solar abundances
    # abundance_to_solar = 1 - abundances / 10**interp.solar_metallicity

    abundance_to_solar = np.c_[abundance_to_solar[:, :-1], abundance_to_solar[:, -2], abundance_to_solar[:, -2], abundance_to_solar[:, -1]] #Add columns for Calcium and Sulphur and add Iron at the end

    #Find helium offsets
    idx_he = find_idx_he(np.log10(abundances[:, 1]), interp.He_bins)
    dx_he = find_dx_he(np.log10(abundances[:, 1]), interp.He_bins, idx_he[:, 0].astype(int))


    # Find redshift offsets
    idx_z = find_idx_z(redshift, interp.redshift_bins)
    dx_z = find_dx_z(redshift, interp.redshift_bins, idx_z[0].astype(int))

    # print('Start interpolation')
    # print(interp.X_Ray.shape, z_index, idx_he[0,:], idx_T[0,:], idx_n[0,:])
    emissivities[joint_mask] = get_table_interp(interp.dn, interp.dT, dx_T, dx_n, idx_T.astype(int), idx_n.astype(int), idx_he.astype(int), dx_he, idx_z.astype(int), dx_z, interp.X_Ray, abundance_to_solar[:, 2:], z_index)

    return emissivities

    