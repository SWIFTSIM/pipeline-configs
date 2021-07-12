"""
Plots the stellar abundances of given snapshot
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
from swiftsimio import load
from swiftpipeline.argumentparser import ScriptArgumentParser



def read_obs_data_OFe(observations_directory):

    # compute COLIBRE standard ratios
    Fe_over_H = 12. - 4.5
    Mg_over_H = 12. - 4.4
    O_over_H = 12. - 3.31
    Mg_over_Fe = Mg_over_H - Fe_over_H
    O_over_Fe = O_over_H - Fe_over_H

    # tabulate/compute the same ratios from Anders & Grevesse (1989)
    Fe_over_H_AG89 = 7.67
    Mg_over_H_AG89 = 7.58
    O_over_H_AG89 = 8.93

    # --
    Mg_over_Fe_AG89 = Mg_over_H_AG89 - Fe_over_H_AG89
    O_over_Fe_AG89 = O_over_H_AG89 - Fe_over_H_AG89

    ## I assume these works use Grevesser & Anders solar metallicity

    file = observations_directory+"/data/StellarAbundances/raw/Letarte_2007.txt"
    data = np.loadtxt(file, skiprows=1)
    FeH_fornax = data[:, 0] + Fe_over_H_AG89 - Fe_over_H
    OFe_fornax = data[:, 4] + O_over_Fe_AG89 - O_over_Fe

    file = observations_directory+"/data/StellarAbundances/raw/Sbordone_2007.txt"
    data = np.loadtxt(file, skiprows=1)
    FeH_sg = data[:, 0] + Fe_over_H_AG89 - Fe_over_H
    OFe_sg = data[:, 4] + O_over_Fe_AG89 - O_over_Fe

    file = observations_directory+"/data/StellarAbundances/raw/Koch_2008.txt"
    data = np.loadtxt(file, skiprows=1)
    FeH_ca = data[:, 0] + Fe_over_H_AG89 - Fe_over_H
    OFe_ca = data[:, 4] + O_over_Fe_AG89 - O_over_Fe

    file = observations_directory+"/data/StellarAbundances/raw/Geisler_2005.txt"
    data = np.loadtxt(file, skiprows=3)
    FeH_scu = data[:, 0] + Fe_over_H_AG89 - Fe_over_H
    OFe_scu = data[:, 4] - data[:, 0] + O_over_Fe_AG89 - O_over_Fe

    # MW data
    FeH_MW = []
    OFe_MW = []

    file = observations_directory+"/data/StellarAbundances/raw/Koch_2008.txt"
    data = np.loadtxt(file, skiprows=3)
    FeH_koch = data[:, 1] + Fe_over_H_AG89 - Fe_over_H
    OH_koch = data[:, 2]
    OFe_koch = OH_koch - FeH_koch + O_over_Fe_AG89 - O_over_Fe

    FeH_MW = np.append(FeH_MW, FeH_koch)
    OFe_MW = np.append(OFe_MW, OFe_koch)

    file = observations_directory+"/data/StellarAbundances/raw/Bai_2004.txt"
    data = np.loadtxt(file, skiprows=3, usecols=[1, 2])
    FeH_bai = data[:, 0] + Fe_over_H_AG89 - Fe_over_H
    OFe_bai = data[:, 1] + O_over_Fe_AG89 - O_over_Fe

    FeH_MW = np.append(FeH_MW, FeH_bai)
    OFe_MW = np.append(OFe_MW, OFe_bai)

    file = observations_directory+"/data/StellarAbundances/raw/Cayrel_2004.txt"
    data = np.loadtxt(file, skiprows=18, usecols=[2, 6])
    FeH_cayrel = data[:, 0] + Fe_over_H_AG89 - Fe_over_H
    OFe_cayrel = data[:, 1] + O_over_Fe_AG89 - O_over_Fe

    FeH_MW = np.append(FeH_MW, FeH_cayrel)
    OFe_MW = np.append(OFe_MW, OFe_cayrel)

    file = observations_directory+"/data/StellarAbundances/raw/Israelian_1998.txt"
    data = np.loadtxt(file, skiprows=3, usecols=[1, 3])
    FeH_isra = data[:, 0] + Fe_over_H_AG89 - Fe_over_H
    OFe_isra = data[:, 1] + O_over_Fe_AG89 - O_over_Fe

    FeH_MW = np.append(FeH_MW, FeH_isra)
    OFe_MW = np.append(OFe_MW, OFe_isra)

    file = observations_directory+"/data/StellarAbundances/raw/Mishenina_1999.txt"
    data = np.loadtxt(file, skiprows=3, usecols=[1, 3])
    FeH_mish = data[:, 0] + Fe_over_H_AG89 - Fe_over_H
    OFe_mish = data[:, 1] + O_over_Fe_AG89 - O_over_Fe

    FeH_MW = np.append(FeH_MW, FeH_mish)
    OFe_MW = np.append(OFe_MW, OFe_mish)

    file = observations_directory+"/data/StellarAbundances/raw/Zhang_Zhao_2005.txt"
    data = np.loadtxt(file, skiprows=3)
    FeH_zhang = data[:, 0] + Fe_over_H_AG89 - Fe_over_H
    OFe_zhang = data[:, 1] + O_over_Fe_AG89 - O_over_Fe

    FeH_MW = np.append(FeH_MW, FeH_zhang)
    OFe_MW = np.append(OFe_MW, OFe_zhang)
    return FeH_MW, OFe_MW, FeH_ca, OFe_ca, FeH_scu, OFe_scu, \
           FeH_fornax, OFe_fornax, FeH_sg, OFe_sg

def read_obs_data_MgFe(observations_directory):
    # -------------------------------------------------------------------------------
    # alpha-enhancement (Mg/Fe), extracted manually from Tolstoy, Hill & Tosi (2009)
    # -------------------------------------------------------------------------------
    file = observations_directory+"/data/StellarAbundances/raw/Fornax.txt"
    data = np.loadtxt(file)
    FeH_fornax = data[:, 0]
    MgFe_fornax = data[:, 1]

    file = observations_directory+"/data/StellarAbundances/raw/Sculptor.txt"
    data = np.loadtxt(file)
    FeH_sculptor = data[:, 0]
    MgFe_sculptor = data[:, 1]

    file = observations_directory+"/data/StellarAbundances/raw/Sagittarius.txt"
    data = np.loadtxt(file)
    FeH_sagittarius = data[:, 0]
    MgFe_sagittarius = data[:, 1]

    file = observations_directory+"/data/StellarAbundances/raw/Carina.txt"
    data = np.loadtxt(file)
    FeH_carina = data[:, 0]
    MgFe_carina = data[:, 1]

    file = observations_directory+"/data/StellarAbundances/raw/MW.txt"
    data = np.loadtxt(file)
    FeH_mw = data[:, 0]
    MgFe_mw = data[:, 1]

    # compute COLIBRE standard ratios
    Fe_over_H = 12. - 4.5
    Mg_over_H = 12. - 4.4
    O_over_H = 12. - 3.31
    Mg_over_Fe = Mg_over_H - Fe_over_H
    O_over_Fe = O_over_H - Fe_over_H

    # tabulate/compute the same ratios from Anders & Grevesse (1989)
    Fe_over_H_AG89 = 7.67
    Mg_over_H_AG89 = 7.58
    O_over_H_AG89 = 8.93
    Mg_over_Fe_AG89 = Mg_over_H_AG89 - Fe_over_H_AG89
    O_over_Fe_AG89 = O_over_H_AG89 - Fe_over_H_AG89

    # correct normalisations for COLIBRE standard
    FeH_fornax += Fe_over_H_AG89 - Fe_over_H
    FeH_sculptor += Fe_over_H_AG89 - Fe_over_H
    FeH_sagittarius += Fe_over_H_AG89 - Fe_over_H
    FeH_carina += Fe_over_H_AG89 - Fe_over_H
    FeH_mw += Fe_over_H_AG89 - Fe_over_H

    MgFe_fornax += Mg_over_Fe_AG89 - Mg_over_Fe
    MgFe_sculptor += Mg_over_Fe_AG89 - Mg_over_Fe
    MgFe_sagittarius += Mg_over_Fe_AG89 - Mg_over_Fe
    MgFe_carina += Mg_over_Fe_AG89 - Mg_over_Fe
    MgFe_mw += Mg_over_Fe_AG89 - Mg_over_Fe
    return FeH_mw, MgFe_mw, FeH_carina, MgFe_carina, FeH_sculptor, MgFe_sculptor, \
           FeH_fornax, MgFe_fornax, FeH_sagittarius, MgFe_sagittarius

def read_data(filename):
    """
    Grabs the data
    """

    mp_in_cgs = 1.6737236e-24
    mH_in_cgs = 1.00784 * mp_in_cgs
    mFe_in_cgs = 55.845 * mp_in_cgs
    mO_in_cgs = 15.999 * mp_in_cgs
    mMg_in_cgs = 24.305 * mp_in_cgs

    # Asplund et al. (2009)
    Fe_H_Sun = 7.5
    O_H_Sun = 8.69
    Mg_H_Sun = 7.6

    O_Fe_Sun = O_H_Sun - Fe_H_Sun - np.log10(mFe_in_cgs / mO_in_cgs)
    Mg_Fe_Sun = Mg_H_Sun - Fe_H_Sun - np.log10(mFe_in_cgs / mMg_in_cgs)
    Fe_H_Sun = Fe_H_Sun - 12.0 - np.log10(mH_in_cgs / mFe_in_cgs)

    data = load(filename)

    oxygen = data.stars.element_mass_fractions.oxygen
    magnesium = data.stars.element_mass_fractions.magnesium
    iron = data.stars.element_mass_fractions.iron
    hydrogen = data.stars.element_mass_fractions.hydrogen

    Fe_H = np.log10(iron / hydrogen) - Fe_H_Sun
    O_Fe = np.log10(oxygen / iron) - O_Fe_Sun
    Mg_Fe = np.log10(magnesium / iron) - Mg_Fe_Sun
    Fe_H[iron == 0] = -7  # set lower limit
    Fe_H[Fe_H < -7] = -7  # set lower limit
    Mg_Fe[iron == 0] = -2  # set lower limit
    Mg_Fe[magnesium == 0] = -2  # set lower limit
    Mg_Fe[Mg_Fe < -2] = -2  # set lower limit
    O_Fe[iron == 0] = -2  # set lower limit
    O_Fe[oxygen == 0] = -2  # set lower limit
    O_Fe[O_Fe < -2] = -2  # set lower limit

    return Fe_H, O_Fe, Mg_Fe

def plot_abundances(snapshot_filename, observations_directory, output_path, name):

    data = load(snapshot_filename)
    redshift = data.metadata.z

    Fe_H, O_Fe, Mg_Fe = read_data(snapshot_filename)

    FeH_MW, OFe_MW, FeH_ca, OFe_ca, FeH_scu, OFe_scu, \
    FeH_fornax, OFe_fornax, FeH_sg, OFe_sg = read_obs_data_OFe(observations_directory)

    # Plot parameters
    params = {
        "font.size": 13,
        "font.family": "Times",
        "text.usetex": True,
        "figure.figsize": (5, 4),
        "figure.subplot.left": 0.12,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.18,
        "figure.subplot.top": 0.95,
        "lines.markersize": 4,
        "lines.linewidth": 2.0,
    }
    plt.rcParams.update(params)

    # Plot the interesting quantities
    plt.figure()

    # Box stellar abundance --------------------------------
    plt.subplot(111)
    plt.grid(True)

    plt.plot(Fe_H, O_Fe, 'o', ms=0.5, color='grey',alpha=0.2)

    plt.plot(FeH_MW, OFe_MW, '+', color='orange', ms=4, label='MW')
    plt.plot(FeH_ca, OFe_ca, 'o', color='crimson', ms=4, label='Carina')
    plt.plot(FeH_scu, OFe_scu, '>', color='khaki', ms=4, label='Sculptor')
    plt.plot(FeH_fornax, OFe_fornax, '<', color='royalblue', ms=4, label='Fornax')
    plt.plot(FeH_sg, OFe_sg, '*', ms=4, color='lightblue', label='Sagittarius')

    bins = np.arange(-7.2, 1, 0.2)
    ind = np.digitize(Fe_H, bins)
    xm = [np.median(Fe_H[ind == i]) for i in range(1, len(bins)) if len(Fe_H[ind == i]) > 10]
    ym = [np.median(O_Fe[ind == i]) for i in range(1, len(bins)) if len(O_Fe[ind == i]) > 10]
    plt.plot(xm, ym, '-', lw=1.5, color='black')

    plt.text(-1.9, 2.6, name+' $z$=%0.2f' % redshift)

    plt.xlabel("[Fe/H]", labelpad=2)
    plt.ylabel("[O/Fe]", labelpad=2)
    plt.axis([-7.2, 2, -2, 3])
    plt.legend(loc=[0.05, 0.02], labelspacing=0.1, handlelength=1.5, handletextpad=0.1, frameon=False, ncol=3,
               columnspacing=0.02)
    plt.savefig(f"{output_path}/"+name+"_FeH_OFe.png", dpi=200)

    ###########################################################################

    FeH_mw, MgFe_mw, FeH_carina, MgFe_carina, FeH_sculptor, MgFe_sculptor, \
    FeH_fornax, MgFe_fornax, FeH_sagittarius, MgFe_sagittarius = read_obs_data_MgFe(observations_directory)

    plt.figure()

    # Box stellar abundance --------------------------------
    plt.subplot(111)
    plt.grid(True)

    plt.plot(Fe_H, Mg_Fe, 'o', ms=0.5, color='grey',alpha=0.2)

    plt.plot(FeH_mw, MgFe_mw, '+', color='orange', ms=4, label='MW')
    plt.plot(FeH_carina, MgFe_carina, 'o', color='crimson', ms=4, label='Carina')
    plt.plot(FeH_sculptor, MgFe_sculptor, '>', color='khaki', ms=4, label='Sculptor')
    plt.plot(FeH_fornax, MgFe_fornax, '<', color='royalblue', ms=4, label='Fornax')
    plt.plot(FeH_sagittarius, MgFe_sagittarius, '*', ms=4, color='lightblue', label='Sagittarius')

    bins = np.arange(-7.2, 1, 0.2)
    ind = np.digitize(Fe_H, bins)
    xm = [np.median(Fe_H[ind == i]) for i in range(1, len(bins)) if len(Fe_H[ind == i]) > 10]
    ym = [np.median(Mg_Fe[ind == i]) for i in range(1, len(bins)) if len(Mg_Fe[ind == i]) > 10]
    plt.plot(xm, ym, '-', lw=1.5, color='black')

    plt.text(-1.9, 2.6, name+' $z$=%0.2f' % redshift)

    plt.xlabel("[Fe/H]", labelpad=2)
    plt.ylabel("[Mg/Fe]", labelpad=2)
    plt.axis([-7.2, 2, -2, 3])
    plt.legend(loc=[0.05, 0.02], labelspacing=0.1, handlelength=1.5, handletextpad=0.1, frameon=False, ncol=3,
               columnspacing=0.02)
    plt.savefig(f"{output_path}/"+name+"_FeH_MgFe.png", dpi=200)


arguments = ScriptArgumentParser(
    description="Creates a metallicity density evolution plot for stars."
)

snapshot_filenames = [
    f"{directory}/{snapshot}"
    for directory, snapshot in zip(arguments.directory_list, arguments.snapshot_list)
]

names = arguments.name_list
output_path = arguments.output_directory
observations_directory = f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}"

for snapshot_filename, name in zip(
    snapshot_filenames, names):

    plot_abundances(snapshot_filename, observations_directory, output_path, name)