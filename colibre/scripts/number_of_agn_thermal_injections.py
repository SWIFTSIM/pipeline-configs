"""
Makes a histogram of the number of AGN thermal injections. Uses the swiftsimio library.
"""

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load

N_bounds = [0.7, 7e5]  # Min and max number of AGN thermal energy injections


def get_data(filename):
    """
    Grabs the data
    """

    data = load(filename)

    num_of_agn_thermal_injections = data.black_holes.number_of_heating_events

    # Take only those BHs that have had at least one AGN thermal injection event
    # (i.e. heated at least one gas particle)
    had_at_least_one_agn = num_of_agn_thermal_injections > 0

    return num_of_agn_thermal_injections[had_at_least_one_agn]


def make_single_image(filenames, names, N_bounds, number_of_simulations, output_path):
    """
    Makes a single histogram of the number of AGN thermal energy injections.
    """

    fig, ax = plt.subplots()

    ax.set_xlabel("Total number of AGN thermal injections")
    ax.set_ylabel("Cumulative number of Black Holes")
    ax.loglog()

    for filename, name in zip(filenames, names):
        N_agn_events = get_data(filename)

        h, bin_edges = np.histogram(
            np.log10(N_agn_events), range=np.log10(N_bounds), bins=250, density=False
        )
        bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bins = 10 ** bins

        # The cumsum is done from right to left (along the X axis)
        ax.plot(bins, np.cumsum(h[::-1])[::-1], label=name)

    ax.legend()
    ax.set_xlim(*N_bounds)

    fig.savefig(f"{output_path}/num_agn_thermal_injections.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(
        description="Histogram showing the cumulative number of BHs with a given total"
        " number of thermal energy injections the black hole has had throughout the"
        " entire simulation"
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
        N_bounds=N_bounds,
        number_of_simulations=arguments.number_of_inputs,
        output_path=arguments.output_directory,
    )
