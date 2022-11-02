import matplotlib.pyplot as plt
import numpy as np
from swiftsimio import load, SWIFTDataset

import unyt
from unyt import mh, cm, Gyr, speed_of_light
import glob
from velociraptor.observations import load_observations
from velociraptor.autoplotter.objects import VelociraptorLine
from velociraptor.tools.mass_functions import create_adaptive_mass_function

# Set the limits of the figure.
x_bounds = [1e40, 1e50]  # erg/s
value_bounds = [1e-11, 0.5]  # Mpc^-3 dex^-1


def get_data(data: SWIFTDataset) -> unyt.unyt_array:

    masses = data.black_holes.subgrid_masses.to("Msun")
    try:
        rad_effs = data.black_holes.radiative_efficiencies
    except AttributeError:
        rad_effs = unyt.unyt_array(
            float(
                data.metadata.parameters["COLIBREAGN:radiative_efficiency"].decode(
                    "utf-8"
                )
            )
            * np.ones_like(masses),
            "dimensionless",
        )

    accr_rates = data.black_holes.accretion_rates.astype(np.float64).to(
        unyt.kg / unyt.s
    )
    luminosities = rad_effs * (speed_of_light ** 2) * accr_rates
    values = luminosities.to(unyt.erg / unyt.s)

    return values


def make_single_image(
    filenames, names, x_bounds, value_bounds, output_path, paths_to_observational_data
):

    fig, ax = plt.subplots()

    ax.set_xlabel(
        "AGN Bolometric Luminosity $L_{\\rm bol}$ $[{\\rm erg}{\\rm \\, s}^{-1}]$"
    )
    ax.set_ylabel(
        "AGN Bol. Lum. Func. $\log_{10}\Phi_{\\rm bol}$ [${\\rm Mpc}^{-3}{\\rm dex}^{-1}$]"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Default value of the redshift of the snapshot
    redshift, delta_z = 0.0, 0.1

    for filename, name in zip(filenames, names):

        snapshot = load(filename)
        luminosities = get_data(snapshot)

        redshift = snapshot.metadata.redshift
        boxsize = snapshot.metadata.boxsize.to("Mpc")
        box_volume = boxsize[0] * boxsize[1] * boxsize[2]

        x_low = x_bounds[0] * unyt.erg / unyt.s
        x_high = x_bounds[1] * unyt.erg / unyt.s
        bin_centers, mass_function, error = create_adaptive_mass_function(
            luminosities, x_low, x_high, box_volume, base_n_bins=25, minimum_in_bin=1
        )

        fill_plot, = ax.plot(bin_centers, mass_function, label=name)
        ax.fill_between(
            bin_centers,
            mass_function - error,
            mass_function + error,
            alpha=0.2,
            facecolor=fill_plot.get_color(),
        )

    for index, path_to_observations in enumerate(paths_to_observational_data):
        observations = load_observations(
            path_to_observations,
            redshift_bracket=[redshift - delta_z, redshift + delta_z],
        )
        for observation in observations:
            observation.plot_on_axes(ax)

    ax.legend()
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*value_bounds)

    fig.savefig(f"{output_path}/AGN_bolometric_luminosity_function.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(description="AGN Bolometric Luminosity Function")

    snapshot_filenames = [
        f"{directory}/{snapshot}"
        for directory, snapshot in zip(
            arguments.directory_list, arguments.snapshot_list
        )
    ]

    plt.style.use(arguments.stylesheet_location)

    paths_to_obs_data = glob.glob(
        f"{arguments.config.config_directory}/"
        f"{arguments.config.observational_data_directory}/"
        f"data/BlackHoleAGNBolometricLuminosityFunction/*.hdf5"
    )

    make_single_image(
        filenames=snapshot_filenames,
        names=arguments.name_list,
        x_bounds=x_bounds,
        value_bounds=value_bounds,
        output_path=arguments.output_directory,
        paths_to_observational_data=paths_to_obs_data,
    )
