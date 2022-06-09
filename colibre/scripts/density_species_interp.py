"""
Makes a plot of Hydrogen species fractions as a function of density, with a co-plot of dust to metal ratio
"""

import matplotlib.pyplot as plt
import numpy as np
import glob

from swiftsimio import load
import h5py as h5

from unyt import mh, cm
from scipy.interpolate import interpn

# Set the limits of the figure.
density_bounds = [10 ** (-5), 10 ** (6.0)]  # nh/cm^3
dustfracs_bounds = [-4, -1]  # In metal mass fraction
min_dustfracs = 10 ** dustfracs_bounds[0]
bins = 64
elements = [
    "Hydrogen",
    "Helium",
    "Carbon",
    "Nitrogen",
    "Oxygen",
    "Neon",
    "Magnesium",
    "Silicon",
    "Sulphur",
    "Calcium",
    "Iron",
    "Oatoms",
]


def get_data(filename, tables, prefix_rho, prefix_T):
    """
    Grabs the data (T in Kelvin and density in mh / cm^3) and interpolates for species fractions..
    """

    data = load(filename)
    number_density = (
        getattr(data.gas, f"{prefix_rho}densities").to_physical() / mh
    ).to(cm ** -3)
    temperature = getattr(data.gas, f"{prefix_T}temperatures").to_physical().to("K")
    masses = data.gas.masses.to_physical().to("Msun")

    # Abundance Mass Fractions
    X = data.gas.element_mass_fractions.hydrogen
    Z = data.gas.metal_mass_fractions

    # Species Mass Fractions
    atomfrac = X * masses * data.gas.species_fractions.HI
    molfrac = 2 * X * masses * data.gas.species_fractions.H2
    ionfrac = X * masses * data.gas.species_fractions.HII

    # Dust Mass Fractions
    dfracs = np.zeros(data.gas.masses.shape)
    dsfrac_dict = {}
    for d in data.metadata.named_columns["DustMassFractions"]:
        dfrac = getattr(data.gas.dust_mass_fractions, d)
        dsfrac_dict[d] = dfrac.astype("float64")
        dfracs += dfrac

    if glob.glob(tables):
        with h5.File(tables, "r") as table_file:
            lrho = np.log10(
                (data.gas.subgrid_physical_densities.to_physical() / mh)
                .to(cm ** -3)
                .value
            )
            lT = np.log10((data.gas.subgrid_temperatures).to_physical().to("K").value)
            lZ = np.log10((data.gas.metal_mass_fractions.value / 0.0127))
            z = np.ones(lrho.size) * data.metadata.redshift

            bin_z = np.array(table_file["TableBins/RedshiftBins"])
            bin_lT = np.array(table_file["TableBins/TemperatureBins"])
            bin_lZ = np.array(table_file["TableBins/MetallicityBins"])
            bin_lrho = np.array(table_file["TableBins/DensityBins"])

            z = np.clip(z, bin_z.min(), bin_z.max())
            lZ = np.clip(lZ, bin_lZ.min(), bin_lZ.max())
            lT = np.clip(lT, bin_lT.min(), bin_lT.max())
            lrho = np.clip(lrho, bin_lrho.min(), bin_lrho.max())

            bins = (bin_z, bin_lT, bin_lZ, bin_lrho)
            vecs = np.column_stack([z, lT, lZ, lrho])

            df = pow(10, np.array(table_file["Tdep/Depletion"]))
            difracs = np.zeros(z.shape)
            for i in range(df.shape[-1]):
                elname = elements[i].lower()
                if hasattr(data.gas.element_mass_fractions, elname):
                    elfrac = getattr(data.gas.element_mass_fractions, elname)
                    interp_elfdust = interpn(
                        bins, df[:, :, :, :, i], vecs, bounds_error=False
                    )
                    indvdust = elfrac * interp_elfdust
                    difracs += indvdust
                else:
                    continue

    else:
        difracs = np.zeros(masses.size)
        print("Cooling Tables not found. continuing without tables")

    # casting to float64 to avoid arcane np.histogram bug(?)
    out_tuple = (
        number_density.value.astype("float64"),
        temperature.value.astype("float64"),
        difracs.astype("float64"),
        dfracs.astype("float64"),
        dsfrac_dict,
        masses.value.astype("float64"),
        molfrac.value.astype("float64"),
        atomfrac.value.astype("float64"),
        ionfrac.value.astype("float64"),
        X.value.astype("float64"),
        Z.value.astype("float64"),
    )

    return out_tuple


def make_hist(filename, density_bounds, bins, tables, prefix_rho="", prefix_T=""):
    """
    Makes the histogram for filename with bounds as lower, higher
    for the bins and "bins" the number of bins along each dimension.

    Also returns the edges for pcolormesh to use.
    """

    density_bins = np.logspace(
        np.log10(density_bounds[0]), np.log10(density_bounds[1]), bins
    )

    ret_tuple = get_data(filename, tables, prefix_rho, prefix_T)
    nH, T, Di, D, DSdict, Mg, mol, atom, ion, X, Z = ret_tuple

    Hd, density_edges = np.histogram(nH, bins=density_bins, weights=D * Mg)

    Hdi, density_edges = np.histogram(nH, bins=density_bins, weights=Di * Mg)

    H_Z, _ = np.histogram(nH, bins=density_bins, weights=Z * Mg)

    Hh2, _ = np.histogram(nH, bins=density_bins, weights=mol)

    Hhi, _ = np.histogram(nH, bins=density_bins, weights=atom)

    Hhii, _ = np.histogram(nH, bins=density_bins, weights=ion)

    H_norm, _ = np.histogram(nH, bins=density_bins, weights=Mg)

    H_h, _ = np.histogram(nH, bins=density_bins, weights=(X * Mg))

    # Avoid div/0
    mask = H_norm == 0.0
    Hd[mask] = None
    Hdi[mask] = None
    Hh2[mask] = None
    Hhi[mask] = None
    Hhii[mask] = None
    H_h[mask] = 1.0
    H_Z[mask] = 1.0
    H_norm[mask] = 1.0

    # construct dictionary of arrays for individual dust species
    Hds = {}
    for k in DSdict.keys():
        Hds[k], _ = np.histogram(nH, bins=density_bins, weights=DSdict[k] * Mg)
        Hds[k][mask] = None
        Hds[k] = np.ma.array((Hds[k] / H_Z).T, mask=mask.T)

    out_tuple = (
        np.ma.array((Hh2 / H_h).T, mask=mask.T),
        np.ma.array((Hhi / H_h).T, mask=mask.T),
        np.ma.array((Hhii / H_h).T, mask=mask.T),
        np.ma.array((Hd / H_Z).T, mask=mask.T),
        np.ma.array((Hdi / H_Z).T, mask=mask.T),
        Hds,
        density_edges,
    )

    return out_tuple


def setup_axes(number_of_simulations: int, prop_type="hydro"):
    """
    Creates the figure and axis object. Creates a grid of a x b subplots
    that add up to at least number_of_simulations.
    """

    sqrt_number_of_simulations = np.sqrt(number_of_simulations)
    horizontal_number = int(np.ceil(sqrt_number_of_simulations))
    # Ensure >= number_of_simulations plots in a grid
    vertical_number = int(np.ceil(number_of_simulations / horizontal_number))

    fig, ax = plt.subplots(
        vertical_number, horizontal_number, squeeze=True, sharex=True, sharey=True
    )

    ax = np.array([ax]) if number_of_simulations == 1 else ax

    if horizontal_number * vertical_number > number_of_simulations:
        for axis in ax.flat[number_of_simulations:]:
            axis.axis("off")

    # Set all valid on bottom row to have the horizontal axis label.
    for axis in np.atleast_2d(ax)[:][-1]:
        if prop_type == "hydro":
            axis.set_xlabel(r"Log Density [$\log_{10}(n_H /\; {\rm cm}^{-3})$]")
        elif prop_type == "subgrid":
            axis.set_xlabel(r"Log Subgrid Density [$\log_{10}(n_H /\; {\rm cm}^{-3})$]")
        else:
            raise Exception("Unrecognised property type.")
    for axis in np.atleast_2d(ax).T[:][0]:
        axis.set_ylabel("Species Mass Fraction")
        axis2 = axis.twinx()
        axis2.set_ylabel("\n\n\n Fraction of Metals in Dust")
        axis2.set_yticks([])
    return fig, ax


def make_single_image(
    filenames,
    names,
    number_of_simulations,
    density_bounds,
    bins,
    output_path,
    prop_type,
    tables,
):
    """
    Makes a single plot of rho-T
    """

    if prop_type == "subgrid":
        prefix_rho = "subgrid_physical_"
        prefix_T = "subgrid_"
    else:
        prefix_rho = ""
        prefix_T = ""

    fig, ax = setup_axes(
        number_of_simulations=number_of_simulations, prop_type=prop_type
    )

    hist_h2 = []
    hist_hi = []
    hist_hii = []
    hist_d2z = []
    hist_di2z = []
    hists_hds2Z = []

    for filename in filenames:
        hh2, hhi, hhii, hd2z, hdi2z, hds2Z_dict, d = make_hist(
            filename, density_bounds, bins, tables, prefix_rho, prefix_T
        )
        hist_h2.append(hh2)
        hist_hi.append(hhi)
        hist_hii.append(hhii)
        hist_d2z.append(hd2z)
        hist_di2z.append(hdi2z)
        hists_hds2Z.append(hds2Z_dict)

    ncols = 20
    collist = []
    binmids = np.log10(d)[:-1] + 0.5 * np.diff(np.log10(d))

    for hist_h2, hist_hi, hist_hii, hist_d2z, hist_di2z, hists_hds2Z, name, axis in zip(
        hist_h2, hist_hi, hist_hii, hist_d2z, hist_di2z, hists_hds2Z, names, ax.flat
    ):

        axis.plot(binmids, np.clip(hist_h2, 0, 1), label="molecular")
        axis.plot(binmids, np.clip(hist_hi, 0, 1), label="atomic")
        axis.plot(binmids, np.clip(hist_hii, 0, 1), label="ionised")
        axis.plot(
            binmids, hist_h2 + hist_hi + hist_hii, color="0.7", label="total", ls=":"
        )
        axis.text(0.025, 0.975, name, ha="left", va="top", transform=axis.transAxes)
        axis.legend(frameon=False, loc=6)
        axis.set_ylim(0, 1.1)
        axis.set_xlim(np.log10(density_bounds[0]), np.log10(density_bounds[1]))
        axis2 = axis.twinx()
        axis2.plot(binmids, hist_d2z, c="C5", label="Model dust")
        axis2.plot(binmids, hist_di2z, c="C5", ls="--", label="Table dust")
        dotcount = 0
        for k in hists_hds2Z.keys():
            axis2.plot(
                binmids,
                hists_hds2Z[k],
                c="C6",
                lw=0.5 * (len(hists_hds2Z.keys()) - dotcount),
                ls=":",
                label=k,
            )
            dotcount += 1
            print(dotcount)
        axis2.legend(frameon=False, loc=(0.02, 0.7))
        axis2.set_ylim(0, axis2.get_ylim()[1])

    fig.savefig(f"{output_path}/{prefix_T}density_vs_species_fraction.png")
    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(
        description="Hydrogen species figure.",
        additional_arguments={
            "quantity_type": "hydro",
            "cooling_tables": "UV_dust1_CR1_G1_shield1.hdf5",
        },
    )

    snapshot_filenames = [
        f"{directory}/{snapshot}"
        for directory, snapshot in zip(
            arguments.directory_list, arguments.snapshot_list
        )
    ]

    plt.style.use(arguments.stylesheet_location)

    make_single_image(
        filenames=snapshot_filenames,
        names=arguments.name_list,
        number_of_simulations=arguments.number_of_inputs,
        density_bounds=density_bounds,
        bins=bins,
        output_path=arguments.output_directory,
        prop_type=arguments.quantity_type,
        tables=arguments.cooling_tables,
    )
