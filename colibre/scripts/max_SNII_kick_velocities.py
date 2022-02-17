"""
Makes a histogram of maximal kicked velocities experienced by the gas particles in SNII
kinetic feedback. Uses the swiftsimio library.
"""

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load

v_kick_bounds = [0.1, 1e5]  # in km/s


def get_data(filename):
    """
    Grabs the data
    """

    data = load(filename)

    # We need to select only only gas but also stars because sparts were gas at some
    # point in the past and hence could be kicked.
    stars_SNII_v_kick_max = (data.stars.maximal_sniikinetic_feedbackvkick).to("km/s")
    gas_SNII_v_kick_max = (data.gas.maximal_sniikinetic_feedbackvkick).to("km/s")

    # Limit only to those gas/stellar particles that were in fact kicked by SNII
    stars_SNII_kicked = stars_SNII_v_kick_max > 0.0
    gas_SNII_kicked = gas_SNII_v_kick_max > 0.0

    # Select only those parts that were kicked by SNII in the past
    stars_SNII_v_kick_max = stars_SNII_v_kick_max[stars_SNII_kicked]
    gas_SNII_v_kick_max = gas_SNII_v_kick_max[gas_SNII_kicked]

    # All kicks (contained in gas + stars info)
    v_kick_max_all = np.concatenate(
        [
            stars_SNII_v_kick_max,
            gas_SNII_v_kick_max,
        ]
    )

    return v_kick_max_all


def make_single_image(
    filenames, names, v_kick_bounds, number_of_simulations, output_path
):
    """
    Makes a single histogram of the maximal SNII kick velocities.
    """

    fig, ax = plt.subplots()

    ax.set_xlabel("Maximal SNII kick velocity $v_{\\rm kick,max}$ [km s$^{-1}$]")
    ax.set_ylabel("PDF [-]")
    ax.loglog()

    for filename, name in zip(filenames, names):
        v_kick_max = get_data(filename)
        h, bin_edges = np.histogram(
            np.log10(v_kick_max), range=np.log10(v_kick_bounds), bins=250, density=True
        )
        bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bins = 10**bins
        ax.plot(bins, h, label=name)

    ax.legend()
    ax.set_xlim(*v_kick_bounds)

    fig.savefig(f"{output_path}/SNII_maximal_kick_velocities.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(
        description="Basic max SNII kick velocity histogram."
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
        v_kick_bounds=v_kick_bounds,
        number_of_simulations=arguments.number_of_inputs,
        output_path=arguments.output_directory,
    )
