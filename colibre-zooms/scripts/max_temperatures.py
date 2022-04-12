"""
Makes a histogram of the gas particle max temperature. Uses the swiftsimio library.
"""

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load

T_bounds = [1e5, 1e12]  # in K


def get_data(filename):
    """
    Grabs the data
    """

    data = load(filename)

    gas_max_T = data.gas.maximal_temperatures.to("K")

    return gas_max_T


def make_single_image(filenames, names, T_bounds, number_of_simulations, output_path):
    """
    Makes a single histogram of the gas particle max temperatures.
    """

    fig, ax = plt.subplots()

    ax.set_xlabel("Gas Max. Temperature $T_{\\rm max}$ [K]")
    ax.set_ylabel("PDF [-]")
    ax.loglog()

    log_max_Ts = []

    for filename, name in zip(filenames, names):
        T_max = get_data(filename)
        log_T_max = np.log10(T_max)
        h, bin_edges = np.histogram(
            log_T_max, range=np.log10(T_bounds), bins=250, density=True
        )
        bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bins = 10 ** bins
        ax.plot(bins, h, label=name)

        log_max_Ts.append(f"{name}: {log_T_max.max():.2f}")

    ax.text(
        0.05,
        0.05,
        "\n".join(["Maximum $\\log_{10} T_{\\rm max} /$K"] + log_max_Ts),
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )

    ax.legend(loc="upper right", markerfirst=False)
    ax.set_xlim(*T_bounds)

    fig.savefig(f"{output_path}/gas_max_temperatures.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(
        description="Basic gas particle max temperature histogram."
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
        T_bounds=T_bounds,
        number_of_simulations=arguments.number_of_inputs,
        output_path=arguments.output_directory,
    )
