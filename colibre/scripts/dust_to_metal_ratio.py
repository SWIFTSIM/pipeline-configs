"""
Makes a histogram of the dust to metal ratio for the cold dense
gas. Uses the swiftsimio library.
"""

import matplotlib.pyplot as plt
import numpy as np
import swiftsimio
import unyt

cold_dense_gas_max_temperature = 10**4.5 * unyt.K
cold_dense_gas_min_hydrogen_number_density = 0.1 / unyt.cm**3
solar_metal_mass_fraction = 0.0134 # Z_sun
twelve_plus_log_OH_solar = 8.69

plot_bounds = [10**-5, 10**0]

def get_data(filename):
    """
    Grabs the data
    """

    data = swiftsimio.load(filename)

    cold_dense_mask = data.gas.temperatures < cold_dense_gas_max_temperature
    hydrogen_number_density = data.gas.densities / unyt.mh
    cold_dense_mask &= hydrogen_number_density > cold_dense_gas_min_hydrogen_number_density

    nH = data.gas.element_mass_fractions.hydrogen.value[cold_dense_mask]
    nO = data.gas.element_mass_fractions.oxygen.value[cold_dense_mask]
    gas_O_over_H = nO / (16.0 * nH)

    metal_frac = np.zeros_like(gas_O_over_H)
    metal_frac[gas_O_over_H > 0.0] = (
        pow(
            10,
            np.log10(
                gas_O_over_H[gas_O_over_H > 0.0]
            )
            + 12
            - twelve_plus_log_OH_solar,
        )
        * solar_metal_mass_fraction
    )

    dust_frac = np.zeros_like(gas_O_over_H)
    for dust_type in data.gas.dust_mass_fractions.named_columns:
        dust_frac += getattr(data.gas.dust_mass_fractions, dust_type).value[cold_dense_mask]

    dust_to_metal = np.zeros_like(gas_O_over_H)
    dust_to_metal[metal_frac > 0.0] = (
        dust_frac[metal_frac > 0.0] / metal_frac[metal_frac > 0.0]
    )

    return dust_to_metal[dust_to_metal != 0]


def make_single_image(
    filenames, names, number_of_simulations, output_path
):
    """
    Makes a single histogram of the dust to metal ratio.
    """

    fig, ax = plt.subplots()

    ax.set_xlabel(f"$\\mathcal{{DTM}}$ (cold, dense phase)")
    ax.set_ylabel("PDF [-]")
    ax.loglog()

    for filename, name in zip(filenames, names):
        dust_to_metal = get_data(filename)
        h, bin_edges = np.histogram(
            np.log10(dust_to_metal), bins=250, density=True
        )
        bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bins = 10 ** bins
        ax.plot(bins, h, label=name)

    ax.legend()
    ax.set_xlim(plot_bounds)

    fig.savefig(f"{output_path}/dust_to_metal_ratio.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(description="Basic dust to metals histogram.")

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
        output_path=arguments.output_directory,
    )
