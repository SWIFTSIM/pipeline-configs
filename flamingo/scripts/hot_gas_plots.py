"""
Creates the plots of hot gas properties
"""

import matplotlib
import sys
if not hasattr(sys, 'ps1'): 
    matplotlib.use('Agg')
import pandas as pd
from astropy import cosmology
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from natsort import natsorted
import latexify
import glob
import os
from swiftsimio import load as load_snap
from velociraptor import load as load_catalogue
from velociraptor.particles import load_groups
from velociraptor.swift.swift import to_swiftsimio_dataset
import multiprocessing
from contextlib import closing
from functools import partial
import interpolate_X_Ray_redshift as xray
from numba import njit
from scipy.interpolate import CubicSpline
import unyt
from swiftpipeline.argumentparser import ScriptArgumentParser

pd.options.mode.chained_assignment = None

sigma_t = 6.6524e-25                        # Thomson cross-section (cm**2)
ergs_keV = 1.60215e-9                       # erg -> keV conversion
kb = 1.3807e-16                             # Boltzmann's constant (erg/K)
mp = 1.6726e-24                             # proton mass (g)
mpc = 3.0857e24                             # Mpc -> cm conversion
mu = 0.6
Zmet = 0.3
G = 6.67408e-08                             # cm3 / (g s2)

z_tab = [0.0, 10 ** -1.5, 10 ** -1, 10 ** -0.5, 1, 10 ** 0.5]
'''
Xe = ne/nH  (see Sutherland & Dopita 1993)
valid over the range 0 <= Z/Zsolar <= 3.16
'''
xe = [1.128, 1.129, 1.131, 1.165, 1.209, 1.238] 
xe_int = CubicSpline(z_tab, xe, bc_type=((2, 0.0), (2, 0.0)), extrapolate=False)
'''
Xi = ni/nH  (see Sutherland & Dopita 1993)
valid over the range 0 <= Z/Zsolar <= 3.16
'''
xi = [1.064, 1.064, 1.064, 1.08, 1.099, 1.103] 
xi_int = CubicSpline(z_tab, xi, bc_type=((2, 0.0), (2, 0.0)), extrapolate=False)

ne_n = xe_int(Zmet) / (xe_int(Zmet) + xi_int(Zmet))

@njit
def compute_Xs_mu(elements_array):

    all_elements = ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'sulphur', 'calcium', 'iron']
    element_masses = [1.00794, 4.002602, 12.0107, 14.0067, 15.9994, 20.1797, 24.3050, 28.0855, 32.065, 40.078, 55.845]
    agtomic_num = [1.0, 2.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 26.0]

    # Hydrogen
    ne_nH = np.full_like(elements_array[:, 0], 1.0)
    ni_nH = np.full_like(elements_array[:, 0], 1.0)
    mu = np.full_like(elements_array[:, 0], 0.5)

    # Every other element
    for i in range(len(all_elements) - 1):
        ne_nH += elements_array[:, i + 1] * (element_masses[0] / element_masses[i + 1]) * (agtomic_num[i + 1] / agtomic_num[0])
        ni_nH += elements_array[:, i + 1] * (element_masses[0] / element_masses[i + 1]) 
        mu += elements_array[:, i + 1] / (agtomic_num[0] + agtomic_num[i + 1])

    return ne_nH, ni_nH, mu

def read_hot_gas_props(run_directory, snapshot_name, cat_name, obs_data_dir, chunk):
    
    elements = ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'iron']
    all_elements = ['hydrogen', 'helium', 'carbon', 'nitrogen', 'oxygen', 'neon', 'magnesium', 'silicon', 'sulphur', 'calcium', 'iron']
    SulphurOverSilicon = 0.6054160
    CalciumOverSilicon = 0.0941736

    catalogue = load_catalogue(run_directory + '/' + cat_name)
    groups = load_groups(run_directory + '/' + cat_name.replace('properties', 'catalog_groups'), catalogue=catalogue)

    df_chunk = pd.DataFrame()
    halo_n = 1
    tot_h = len(chunk.index)
    for index, row in chunk.iterrows():
        print('TaskID: ', multiprocessing.current_process()._identity[0], 'Processing halo = ', int(row.GroupNumber), '(', halo_n, '/', tot_h, ')')
        particles, unbound_particles = groups.extract_halo(halo_id=int(row.GroupNumber))
        data, bmask = to_swiftsimio_dataset(particles, run_directory + '/' + snapshot_name, generate_extra_mask=True)
        udata, umask = to_swiftsimio_dataset(unbound_particles, run_directory + '/' + snapshot_name, generate_extra_mask=True)
        mask = np.logical_or(bmask.gas, umask.gas)

        meta = data.metadata        
        BoxSize = meta.boxsize[0].value
        cosmo =  meta.cosmology
        aexp = meta.a
        z = meta.redshift
        ez = cosmo.efunc(z)

        df_elements = pd.DataFrame()
        for element in elements:
            df_elements[element] = getattr(data.gas.smoothed_element_mass_fractions, element)

        df_gas = pd.DataFrame(data.gas.coordinates.value, columns=['x', 'y', 'z'])
        df_gas['temp'] = data.gas.temperatures.value
        df_gas['mass'] = data.gas.masses.to('Msun').value.astype(np.float64) * u.Msun.to(u.gram)
        # df_gas['metallicity'] = data.gas.metal_mass_fractions.value / 0.0134
        df_gas['sfr'] = data.gas.star_formation_rates.to('Msun/Gyr').value
        df_gas['rho'] = data.gas.densities.to('g * cm ** -3').value
        df_gas['nH'] = df_elements.hydrogen.values * df_gas.rho.values / mp 
        # df_gas['nH'] = np.full_like(df_gas.temp.values, 1e6)
        df_gas['lambda_c'] = np.power(10, xray.interpolate_X_Ray_pandas(np.log10(df_gas.nH.values), np.log10(df_gas.temp.values), df_elements, z, obs_data_dir, band = '0.5-2.0keV', fill_value = -400)) / df_gas.nH.values ** 2
        df_gas['in_FOF'] = mask

        df_elements = df_elements.div(df_elements.hydrogen, axis=0)
        df_elements['sulphur'] = df_elements.silicon * SulphurOverSilicon
        df_elements['calcium'] = df_elements.silicon * CalciumOverSilicon

        df_elements = df_elements.reindex(columns=all_elements)
        ne_nH, ni_nH, mu = compute_Xs_mu(df_elements.to_numpy())
        
        df_gas['ne_nH'] = ne_nH
        df_gas['ni_nH'] = ni_nH
        df_gas['mu'] = mu

        df_gas['GroupNumber'] = int(row.GroupNumber)
        df_gas['m500'] = row.m500
        df_gas['r500'] = row.r500
        df_gas['CoP_x'] = row.CoP_x
        df_gas['CoP_y'] = row.CoP_y
        df_gas['CoP_z'] = row.CoP_z
        df_gas['S500'] = row.S500
        df_gas['P500'] = row.P500

        df_gas['x'] = df_gas.x - df_gas.CoP_x + BoxSize / 2
        df_gas['y'] = df_gas.y - df_gas.CoP_y + BoxSize / 2
        df_gas['z'] = df_gas.z - df_gas.CoP_z + BoxSize / 2
        df_gas.x.loc[df_gas.x < 0] = df_gas.x.loc[df_gas.x < 0] + BoxSize
        df_gas.y.loc[df_gas.y < 0] = df_gas.y.loc[df_gas.y < 0] + BoxSize
        df_gas.z.loc[df_gas.z < 0] = df_gas.z.loc[df_gas.z < 0] + BoxSize
        df_gas.x.loc[df_gas.x > BoxSize] = df_gas.x.loc[df_gas.x > BoxSize] - BoxSize
        df_gas.y.loc[df_gas.y > BoxSize] = df_gas.y.loc[df_gas.y > BoxSize] - BoxSize
        df_gas.z.loc[df_gas.z > BoxSize] = df_gas.z.loc[df_gas.z > BoxSize] - BoxSize               
        df_gas['distance'] = (((BoxSize / 2) - df_gas.x) ** 2 + ((BoxSize / 2) - df_gas.y) ** 2 + ((BoxSize / 2) - df_gas.z) ** 2) ** 0.5
        df_gas['distance'] = df_gas.distance / df_gas.r500
        df_gas = df_gas.loc[df_gas.distance <= 5]

        df_gas['Lx'] = df_gas.lambda_c * (df_gas.rho * (df_gas.ne_nH / ((df_gas.ne_nH + df_gas.ni_nH) * df_gas.mu * mp)).values.astype(np.float64) ** 2) * df_gas.mass.values / df_gas.ne_nH
        # df_gas['Lx'] = df_gas.lambda_c * df_gas.mass.values / df_gas.rho

        df_gas['ypar'] = (sigma_t / (511 * ergs_keV)) * kb * df_gas.temp * (df_gas.mass.values * 0.752 * df_gas.ne_nH / mp ) / mpc / mpc # in Mpc**2
        df_gas['ypar'] = df_gas.ypar * (ez ** (-2./3)) 

        df_chunk = pd.concat([df_chunk, df_gas])
        halo_n += 1

    return(df_chunk)

def plot_hot_gas_props(arguments):
    # print('\033[34m TEST... \033[0m')

    run_names = arguments.name_list
    run_directories = [f"{directory}" for directory in arguments.directory_list]
    snapshot_names = [f"{snapshot}" for snapshot in arguments.snapshot_list]
    cat_names = [f"{snapshot}" for snapshot in arguments.catalogue_list]
    obs_data_dir = arguments.config_directory + '/../observational_data/data/HotGasProps/'
    output_path = arguments.output_directory    

    # bahamas_true_Lx = pd.read_csv(obs_data_dir + 'Bahamas_LX_true.csv')
    # bahamas_obs_Lx = pd.read_csv(obs_data_dir + 'Bahamas_LX.csv')

    bahamas_LX = pd.read_csv(obs_data_dir + 'new_BAHAMAS_LX.csv')

    # bahamas_obs_tSZ = pd.read_csv(obs_data_dir + 'Bahamas_tSZ.csv')
    # bahamas_true_tSZ = pd.read_csv(obs_data_dir + 'Bahamas_Ysz_true.csv') 

    bahamas_Ysz = pd.read_csv(obs_data_dir + 'new_BAHAMAS_Ysz.csv')
    planck_tSZ = pd.read_csv(obs_data_dir + 'Planck_tSZ.csv')
    

    # bahamas_obs_rho = pd.read_csv(obs_data_dir + 'Bahamas_rhox.csv')
    # bahamas_obs_P = pd.read_csv(obs_data_dir + 'BAHAMAS_P.csv')  

    bahamas_rho = pd.read_csv(obs_data_dir + 'new_BAHAMAS_rho.csv')
    bahamas_P = pd.read_csv(obs_data_dir + 'new_BAHAMAS_P.csv') 
    bahamas_S_S500 = pd.read_csv(obs_data_dir + 'new_BAHAMAS_S_S500.csv')


    sun_ne = pd.read_csv(obs_data_dir + 'Sun2009_ne_profiles.txt', skiprows=4, delim_whitespace=True, names=['log_r', 'log_ne', 'log_err_m', 'log_err_p'])
    sun_P = pd.read_csv(obs_data_dir + 'Sun2011_P_profiles.txt', skiprows=4, delim_whitespace=True, names=['log_r', 'log_P', 'log_err_m', 'log_err_p'])
    sun_S = pd.read_csv(obs_data_dir + 'Sun09_entropy_profiles.txt')
    ras_S = pd.read_csv(obs_data_dir + 'Rasmussen_entropy_profiles.txt')

    r_bins = np.linspace(-2, np.log10(4), 30)
    centers = (r_bins[:-1] + r_bins[1:]) / 2.
    r_bins = 10 ** r_bins
    centers = 10 ** centers

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    fig6, ax6 = plt.subplots()

    for run_name, run_directory, snapshot_name, cat_name in zip(
        run_names, run_directories, snapshot_names, cat_names):
        # print('[run_name]:', run_name)
        # print('[run_directory]:', run_directory)
        # print('[snapshot_name]:', snapshot_name)
        # print('[cat_name]:', cat_name)
    
        leg = run_name
        try:
            snap = load_snap(run_directory + '/' + snapshot_name)
            catalogue = load_catalogue(run_directory + '/' + cat_name)

            meta = snap.metadata        
            BoxSize = meta.boxsize[0].value
            cosmo =  meta.cosmology
            aexp = meta.a
            z = meta.redshift
            rho_crit = cosmo.critical_density(meta.z).value
            ez = cosmo.efunc(z)
            Ob = meta.cosmology.Ob(z)
            Om = meta.cosmology.Om(z)
            h = meta.cosmology.h
            fb = Ob / Om
            
            hse_bias = 0.8
            print('Reading halos')
            df_halo = pd.DataFrame()
            df_halo['GroupNumber'] = catalogue.ids.id - 1                                                                       # Halo IDs
            df_halo['structuretype'] = catalogue.structure_type.structuretype                                                   # Type of structure
            df_halo['m500'] = catalogue.spherical_overdensities.mass_500_rhocrit.to('Msun')                                     # M500 in Msun
            df_halo['r500'] = catalogue.spherical_overdensities.r_500_rhocrit.to('Mpc') / aexp                                  # co-moving r500 in Mpc
            df_halo['m500'] = df_halo.m500 * hse_bias
            df_halo['r500'] = df_halo.r500 * hse_bias ** (1. / 3)
            df_halo['CoP_x'] = catalogue.positions.xcminpot.to('Mpc') / aexp                                                    # co-moving coordinates in Mpc
            df_halo['CoP_y'] = catalogue.positions.ycminpot.to('Mpc') / aexp                                                    # co-moving coordinates in Mpc  
            df_halo['CoP_z'] = catalogue.positions.zcminpot.to('Mpc') / aexp                                                    # co-moving coordinates in Mpc
            df_halo['kT500'] = G * df_halo.m500 * u.Msun.to(u.gram) * mu * mp / (2 * df_halo.r500 * aexp * u.Mpc.to(u.cm)) / ergs_keV  # keV
            df_halo['rho500'] = 500 * rho_crit * fb                                                                             # g/cm^3
            df_halo['Pe500'] = df_halo.kT500 * ergs_keV * ne_n * (df_halo.rho500 / (mu * mp))                                   # ergs/cm^3
            df_halo['ne500'] = df_halo.Pe500 / (df_halo.kT500 * ergs_keV)                                                       # cm^-3
            df_halo['S500'] = df_halo.kT500 / (df_halo.ne500) ** (2./3.)                                                        # keV cm^2
            df_halo['P500'] = df_halo.kT500 * (df_halo.ne500)                                                                   # keV cm^-3

            df_halo = df_halo.loc[df_halo.m500 >= 1e13]
            df_halo = df_halo.loc[df_halo.structuretype == 10]
            df_halo.reset_index(inplace=True, drop=True)
            # print(df_halo)

            n_halos = len(df_halo.index)
     
            num_processes = 16

            if n_halos < num_processes:
                num_processes = n_halos

            # calculate the chunk size as an integer
            chunk_size = int(np.ceil(n_halos/num_processes))         

            # will work even if the length of the dataframe is not evenly divisible by num_processes
            chunks = [df_halo.iloc[df_halo.index[j:j + chunk_size]] for j in range(0, n_halos, chunk_size)]

            print('Reading gas particles')
            with closing(multiprocessing.Pool(processes=num_processes)) as pool:
                func = partial(read_hot_gas_props, run_directory, snapshot_name, cat_name, obs_data_dir)
                result = pool.map(func, chunks)
                df_all = pd.concat(result, ignore_index=True)

            df_all.sort_values(by=['GroupNumber', 'distance'], inplace=True)

            df_all = df_all.loc[df_all.temp >= 1e5]
            df_all = df_all.loc[df_all.sfr <= 0]

            bin_centers = np.arange(12.8, 15.4, 0.2)
            bin_edges = (bin_centers[1:] + bin_centers[:-1]) / 2.
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.

            # Ysz flux
            inside = df_all.loc[df_all.distance <= 5]
            groups = inside.groupby(['GroupNumber', 'm500'], as_index=False)

            halos = groups.ypar.sum()
            medians, bin_edges, binnumber = stats.binned_statistic(np.log10(halos.m500.values), np.log10(halos.ypar.values), statistic='median', bins=bin_edges)
            n, bin_edges, binnumber = stats.binned_statistic(np.log10(halos.m500.values), np.log10(halos.ypar.values), statistic='count', bins=bin_edges)
            per10, bin_edges, binnumber = stats.binned_statistic(np.log10(halos.m500.values), np.log10(halos.ypar.values), statistic=lambda y: np.percentile(y, 10), bins=bin_edges)
            per90, bin_edges, binnumber = stats.binned_statistic(np.log10(halos.m500.values), np.log10(halos.ypar.values), statistic=lambda y: np.percentile(y, 90), bins=bin_edges)
            mask = np.where(n >= 1)    
            
            ax2.plot(bin_centers[mask], medians[mask], lw=2, label=leg)

            # df_all = df_all.loc[df_all.in_FOF == True]
            # X-ray Luminosity
            inside = df_all.loc[df_all.distance <= 1]
            groups = inside.groupby(['GroupNumber', 'm500'], as_index=False)

            halos = groups.Lx.sum()
            medians, bin_edges, binnumber = stats.binned_statistic(np.log10(halos.m500.values), np.log10(halos.Lx.values), statistic='median', bins=bin_edges)
            n, bin_edges, binnumber = stats.binned_statistic(np.log10(halos.m500.values), np.log10(halos.Lx.values), statistic='count', bins=bin_edges)
            per10, bin_edges, binnumber = stats.binned_statistic(np.log10(halos.m500.values), np.log10(halos.Lx.values), statistic=lambda y: np.percentile(y, 10), bins=bin_edges)
            per90, bin_edges, binnumber = stats.binned_statistic(np.log10(halos.m500.values), np.log10(halos.Lx.values), statistic=lambda y: np.percentile(y, 90), bins=bin_edges)
            mask = np.where(n >= 1)    

            ax1.plot(bin_centers[mask], medians[mask], lw=2, alpha=1, zorder=10, label=leg)

            # Density and Entropy profiles
            
            inside = df_all.loc[(df_all.m500 >= 5.25e13) & (df_all.m500 <= 2e14)]
            # inside = df_all.loc[(df_all.m500 >= 5.75e13) & (df_all.m500 <= 1.3e14)]
            # inside = df_all.loc[(df_all.m500 >= 6e13) & (df_all.m500 <= 1.17e14)]

            print('median = %.2e' % inside.m500.median())
            
            #old
            # inside = df_all.loc[(df_all.m500 >= 5.25e13) & (df_all.m500 <= 2e14)]
            # inside = df_all.loc[(df_all.m500 >= 5.75e13) & (df_all.m500 <= 1.5e14)]
                
            full_index = np.arange(len(inside.GroupNumber.unique()))


            df_rho = pd.DataFrame()
            df_S = pd.DataFrame()
            df_S_S500 = pd.DataFrame()
            df_P = pd.DataFrame()
            for j in range(len(centers)):
                v_j = 'v_' + str(j)
                df_i = inside.copy()
                df_i = df_i.loc[(df_i.distance >= r_bins[j]) & (df_i.distance <= r_bins[j + 1])]
                df_i['volume'] = (4./3) * np.pi * (((df_i.r500 * aexp * u.Mpc.to(u.cm) * r_bins[j + 1]) ** 3.) - ((df_i.r500 * aexp * u.Mpc.to(u.cm) * r_bins[j]) ** 3.))
                df_i['rho_tot'] = df_i.mass / df_i.volume 
                df_i['x_i'] = df_i.ne_nH / ((df_i.ne_nH + df_i.ni_nH) * df_i.mu * mp)
                df_i['ne'] = df_i.rho_tot * df_i.x_i
                df_i['wtemp'] = df_i.temp * df_i.mass
                groups = df_i.groupby(['GroupNumber', 'm500'])
                kT = kb / ergs_keV * groups.wtemp.sum() / groups.mass.sum()
                p = kT * groups.ne.sum()
                s = kT / groups.ne.sum() ** (2./3)
                p /= groups.head(1).P500.median()
                df_S[v_j] = s.reset_index(drop=True).reindex(full_index).values
                s /= groups.head(1).S500.median()
                df_S_S500[v_j] = s.reset_index(drop=True).reindex(full_index).values
                df_rho[v_j] = groups.rho_tot.sum().reset_index(drop=True).reindex(full_index).values / rho_crit
                df_P[v_j] = p.reset_index(drop=True).reindex(full_index).values

            ax3.plot(centers, df_rho.median().values * (centers ** 2.), lw=2, label=leg)
            # ax3.fill_between(centers, df_rho.quantile(.14).values * (centers ** 2.), df_rho.quantile(.84).values * (centers ** 2.), alpha=0.3)

            ax4.plot(centers, df_S.median().values, lw=2, label=leg)
            # ax4.fill_between(centers, df_S.quantile(.14).values, df_S.quantile(.84).values, alpha=0.3)

            ax5.plot(centers, df_S_S500.median().values, lw=2, label=leg)
            # ax5.fill_between(centers, df_S_S500.quantile(.14).values, df_S_S500.quantile(.84).values, alpha=0.3)

            ax6.plot(centers, df_P.median().values * (centers ** 2.), lw=2, label=leg)
            
        except:
            pass

    # ax1.plot(bahamas_obs_Lx.x, bahamas_obs_Lx.y, c='C4', label='$\mathrm{BAHAMAS}_\mathrm{obs}$')
    # ax1.plot(bahamas_true_Lx.x, bahamas_true_Lx.y, c='C9', label='$\mathrm{BAHAMAS}_\mathrm{true}$')
    ax1.plot(bahamas_LX.x, bahamas_LX.y, c='C4', label='BAHAMAS')
    ax1.fill_between(bahamas_LX.x, bahamas_LX.ymin, bahamas_LX.ymax, facecolor='C4', alpha=0.3)
    ax1.set_xlabel('$\mathrm{log}_{10} (M_{500, \mathrm{hse}} \,\, [\mathrm{M}_\odot])$')
    ax1.set_ylabel('$\mathrm{log}_{10} (L_{0.5-2.0 \mathrm{keV, hse}} \,\, [\mathrm{erg/s}])$')
    ax1.set_xlim(13, 15)
    ax1.set_ylim(41, 45.5)
    ax1.set_xticks(np.arange(13, 15.1, 0.5))
    ax1.set_yticks(np.arange(41, 45.6, 1))
    ax1.legend(frameon=False, loc=2)
    fig1.savefig(f'{output_path}/hgas_Lx.png')  

    # ax2.plot(bahamas_obs_tSZ.x, bahamas_obs_tSZ.y, c='C4', label='BAHAMAS$_\mathrm{obs}$')
    # ax2.plot(bahamas_true_tSZ.x, bahamas_true_tSZ.y, c='C9', label='BAHAMAS$_\mathrm{true}$')
    ax2.plot(bahamas_Ysz.x, bahamas_Ysz.y, c='C4', label='BAHAMAS')
    ax2.fill_between(bahamas_Ysz.x, bahamas_Ysz.ymin, bahamas_Ysz.ymax, facecolor='C4', alpha=0.3)
    ax2.errorbar(planck_tSZ.x, planck_tSZ.y, yerr=[planck_tSZ.y - planck_tSZ.ymin, planck_tSZ.ymax - planck_tSZ.y], marker='o', ls='None', lw=1, capsize=2, ms=5, c='k', label='Planck (2015)')  
    ax2.set_xlabel('$\mathrm{log}_{10} (M_{500, \mathrm{hse}} \,\, [\mathrm{M}_\odot])$')
    ax2.set_ylabel('$\mathrm{log}_{10} (\mathrm{E}(z)^{-2/3} Y_\mathrm{SZ, hse}(<5r_{500}) D_A^2 \,\, [\mathrm{Mpc}^2])$')
    ax2.set_xlim(13, 15)
    ax2.set_ylim(-6.5, -3.5)
    ax2.legend(frameon=False, loc=2)
    fig2.savefig(f'{output_path}/hgas_tSZ.png')  

    # ax3.plot(bahamas_obs_rho.x, bahamas_obs_rho.y, c='C4', label='$\mathrm{BAHAMAS}_\mathrm{obs}$')
    ax3.plot(bahamas_rho.x, bahamas_rho.y, c='C4', label='BAHAMAS')
    ax3.fill_between(bahamas_rho.x, bahamas_rho.ymin, bahamas_rho.ymax, facecolor='C4', alpha=0.3)
    x_rho = 10 ** sun_ne.iloc[:-2].log_r
    y_rho = 10 ** sun_ne.iloc[:-2].log_ne * mu * mp / ne_n / rho_crit * x_rho ** 2
    y_rho1 = 10 ** sun_ne.iloc[:-2].log_err_m * mu * mp / ne_n / rho_crit * x_rho ** 2
    y_rho2 = 10 ** sun_ne.iloc[:-2].log_err_p * mu * mp / ne_n / rho_crit * x_rho ** 2 
    ax3.errorbar(x_rho[::3], y_rho[::3], yerr=[y_rho[::3] - y_rho1[::3], y_rho2[::3] - y_rho[::3]], marker='o', ls='None', lw=1, capsize=2, ms=5, c='k', label='Sun+ (2009)')  
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlim(.03,3)
    ax3.set_ylim(0.5,100) 
    ax3.set_xlabel('$r/r_{500, \mathrm{hse}}$')
    ax3.set_ylabel('$\\rho/\\rho_\mathrm{crit} \,\, (r/r_{500, \mathrm{hse}})^2$')
    ax3.legend(frameon=False, loc=2)
    fig3.savefig(f'{output_path}/hgas_rho_r.png')  


    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlim(.03,3)
    ax4.set_ylim(3e1,2e3)   
    ax4.set_xlabel('$r/r_{500, \mathrm{hse}}$')
    ax4.set_ylabel('$S \,\, [\mathrm{keV} \, \mathrm{cm}^2]$')
    ax4.legend(frameon=False, loc=2)
    fig4.savefig(f'{output_path}/hgas_S_r.png')  

    ax5.errorbar(sun_S.r_r500, sun_S.S_S500_50, yerr=[sun_S.S_S500_50 - sun_S.S_S500_10, sun_S.S_S500_90 - sun_S.S_S500_50], marker='o', ls='None', lw=1, capsize=2, ms=5, c='k', label='Sun+ (2009)')  
    ax5.errorbar(ras_S.r_r500, ras_S.S_S500_50, yerr=[ras_S.S_S500_50 - ras_S.S_S500_10, ras_S.S_S500_90 - ras_S.S_S500_50], marker='s', ls='None', lw=1, capsize=2, ms=5, c='k', label='Rasmussen')  

    ax5.plot(bahamas_S_S500.x, bahamas_S_S500.y, c='C4', label='BAHAMAS')
    ax5.fill_between(bahamas_S_S500.x, bahamas_S_S500.ymin, bahamas_S_S500.ymax, facecolor='C4', alpha=0.3)
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.set_xlim(.03,3)
    ax5.set_ylim(1e-1,10)   
    ax5.set_xlabel('$r/r_{500, \mathrm{hse}}$')
    ax5.set_ylabel('$S / S_{500, \mathrm{hse}}$')
    ax5.legend(frameon=False, loc=2)
    fig5.savefig(f'{output_path}/hgas_S_S500_r.png')  

    x_P = 10 ** sun_P.log_r
    y_P = 10 ** sun_P.log_P / (fb / 0.175) * (0.73 / h) ** (8./3.) 

    err_P_m = (y_P - 10 ** sun_P.log_err_m / (fb / 0.175) * (0.73 / h) ** (8./3.)) * x_P ** 2
    err_P_p = (10 ** sun_P.log_err_p / (fb / 0.175) * (0.73 / h) ** (8./3.) - y_P) * x_P ** 2

    ax6.errorbar(x_P[1::3], y_P[1::3] * x_P[1::3] ** 2, yerr=[err_P_m[1::3], err_P_p[1::3]], marker='o', ls='None', lw=1, capsize=2, ms=5, c='k', label='Sun+ (2011)')  
    ax6.plot(bahamas_P.x, bahamas_P.y, c='C4', label='BAHAMAS')
    ax6.fill_between(bahamas_P.x, bahamas_P.ymin, bahamas_P.ymax, facecolor='C4', alpha=0.3)

    ax6.set_xscale('log')
    ax6.set_yscale('log')
    ax6.set_xlim(.03,3)
    ax6.set_ylim(3e-3,1)   
    ax6.set_xlabel('$r/r_{500, \mathrm{hse}}$')
    ax6.set_ylabel('$P / P_{500, \mathrm{hse}} \,\, (r/r_{500, \mathrm{hse}})^2$')
    ax6.legend(frameon=False, loc=2)
    fig6.savefig(f'{output_path}/hgas_P_P500_r.png')  


arguments = ScriptArgumentParser(
    description="Creates hot has plots"
)

plt.style.use(arguments.stylesheet_location)

plot_hot_gas_props(arguments)