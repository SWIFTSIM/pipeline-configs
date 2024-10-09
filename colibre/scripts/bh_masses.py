"""
Makes a BH dyn. mass vs. subgrid mass plot. Uses the swiftsimio library.
"""

import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load

from unyt import unyt_quantity, unyt_array

mass_bounds = [8e2, 2e9]


def get_data(filename):
    """
    Grabs the data (masses in Msun).
    """

    data = load(filename)

    try:
        mass_sub = data.black_holes.subgrid_masses.to("Msun")
        mass_dyn = data.black_holes.dynamical_masses.to("Msun")

    # In case no BHs are found
    except AttributeError:
        mass_sub = unyt_array([], units="Msun")
        mass_dyn = unyt_array([], units="Msun")

    # Fetch gas particle mass
    mass_gas = data.metadata.initial_mass_table.gas.to("Solar_Mass")

    # Fetch BH seed mass
    mass_seed = unyt_quantity(
        float(data.metadata.parameters.get("COLIBREAGN:subgrid_seed_mass_Msun", 0.0)),
        "Msun",
    )

    return mass_sub, mass_dyn, mass_seed, mass_gas


def setup_axes(mass_bounds, number_of_simulations: int):
    """
    Creates the figure and axis object. Creates a grid of a x b subplots
    that add up to at least number_of_simulations.
    """

    sqrt_number_of_simulations = np.sqrt(number_of_simulations)
    horizontal_number = int(np.ceil(sqrt_number_of_simulations))
    # Ensure >= number_of_simulations plots in a grid
    vertical_number = int(np.ceil(number_of_simulations / horizontal_number))

    fig_w, fig_h = plt.figaspect(vertical_number / horizontal_number)
    fig, ax = plt.subplots(
        vertical_number,
        horizontal_number,
        squeeze=True,
        sharex=True,
        sharey=True,
        figsize=(fig_w, fig_h),
    )

    ax = np.array([ax]) if number_of_simulations == 1 else ax

    if horizontal_number * vertical_number > number_of_simulations:
        for axis in ax.flat[number_of_simulations:]:
            axis.axis("off")

    # Set all valid on bottom row to have the horizontal axis label.
    for axis in np.atleast_2d(ax)[:][-1]:
        axis.set_xlabel("BH Dyn. Masses $M_{\\rm dyn}$ [M$_\\odot$]")
        axis.set_xlim(mass_bounds)

    for axis in np.atleast_2d(ax).T[:][0]:
        axis.set_ylabel("BH Sub. Masses $M_{\\rm sub}$ [M$_\\odot$]")
        axis.set_ylim(mass_bounds)

    ax.flat[0].loglog()

    return fig, ax


def make_single_image(
    filenames, names, mass_bounds, number_of_simulations, output_path
):
    """
    Makes a single plot of dynamical vs. subgrid mass for BHs
    """

    fig, ax = setup_axes(mass_bounds, number_of_simulations=number_of_simulations)

    for filename, name, axis in zip(filenames, names, ax.flat):
        m_sub, m_dyn, m_seed, m_gas = get_data(filename)

        # Draw x=y line
        axis.plot(
            [mass_bounds[0], mass_bounds[1]],
            [mass_bounds[0], mass_bounds[1]],
            "k--",
            lw=0.2,
        )

        # Indicate BH seed mass
        axis.plot([mass_bounds[0], mass_bounds[1]], [m_seed, m_seed], "k--", lw=0.2)

        # Indicate gas mass resolution of the run
        axis.plot([m_gas, m_gas], [mass_bounds[0], mass_bounds[1]], "k--", lw=0.2)
        axis.scatter(m_dyn, m_sub, s=1)
        axis.text(
            0.025,
            0.975,
            name,
            ha="left",
            va="top",
            transform=axis.transAxes,
            fontsize=5,
            in_layout=False,
        )

    fig.savefig(f"{output_path}/bh_masses.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(description="Basic BH mass - subgrid mass plot.")

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
        mass_bounds=mass_bounds,
        number_of_simulations=arguments.number_of_inputs,
        output_path=arguments.output_directory,
    )
