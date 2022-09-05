import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from swiftsimio import load

import unyt
from unyt import mh, cm, Gyr, speed_of_light
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
import glob
from velociraptor.observations import load_observation
from pylab import rcParams

# Set the limits of the figure.
x_bounds = [1e20, 1e27]
value_bounds = [3e-9, 5e-3]
bins = 20
def_value = -1.0


def get_data(filename):

    data = load(filename)

    masses = data.black_holes.subgrid_masses.to("Msun")

    try:
        jet_effs = data.black_holes.jet_efficiencies
        accr_rates = data.black_holes.accretion_rates.to(unyt.kg / unyt.s)
        jet_powers = jet_effs * speed_of_light ** 2 * accr_rates

        # Convert from kinetic jet power to radio luminosity using relation from Heckmann & Best (2014)
        f_cav = 4
        radio_luminosities = 1e25 * (jet_powers / unyt.Watt / 7e35 / f_cav) ** (
            1 / 0.68
        )
        values = radio_luminosities * unyt.Watt / unyt.Hertz
    except:
        values = np.zeros(np.size(masses))

    return values


def calculate_medians(filename, x_bounds, value_bounds, bins):

    values = get_data(filename)

    snapshot = load(filename)
    boxsize = snapshot.metadata.boxsize.to("Mpc")
    box_volume = boxsize[0] * boxsize[1] * boxsize[2]
    values = np.log10(values)

    x_bins = np.linspace(np.log10(x_bounds[0]), np.log10(x_bounds[1]), bins)

    bin_width = (np.log10(x_bounds[1]) - np.log10(x_bounds[0])) / bins

    lum_func = []
    lum_func_errors = []
    for x in x_bins:
        values_sliced = values[
            (values > (x - bin_width * 0.5)) & (values < (x + bin_width * 0.5))
        ]

        lum_func.append(np.size(values_sliced) / box_volume)
        lum_func_errors.append(np.sqrt(np.size(values_sliced)) / box_volume)

    return (
        x_bins,
        np.log10(lum_func),
        np.log10(np.array(lum_func) + np.array(lum_func_errors))
        - np.log10(np.array(lum_func)),
    )


def make_single_image(
    filenames,
    names,
    x_bounds,
    value_bounds,
    number_of_simulations,
    output_path,
    observational_data,
):

    fig, ax = plt.subplots()

    ax.set_xlabel("AGN Radio Luminosity $\log_{10}P_{\\rm 1.4GHz}$ $[{\\rm WHz^{-1}}]$")
    ax.set_ylabel(
        "AGN Radio Lum. Func. $\log_{10}\Phi_{\\rm 1.4GHz}$ [${\\rm Mpc}^{-3}{\\rm dex}^{-1}$]"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")

    for filename, name in zip(filenames, names):
        x_bins, luminosity_function, luminosity_function_errors = calculate_medians(
            filename, x_bounds, value_bounds, bins
        )
        mask = luminosity_function > -100
        fill_plot, = ax.plot(
            10 ** x_bins[mask], 10 ** luminosity_function[mask], label=name
        )
        ax.fill_between(
            10 ** x_bins[mask],
            10 ** (luminosity_function[mask] - luminosity_function_errors[mask]),
            10 ** (luminosity_function[mask] + luminosity_function_errors[mask]),
            alpha=0.2,
            facecolor=fill_plot.get_color(),
        )

    rcParams.update({"lines.markersize": 5})
    for index, observation in enumerate(observational_data):
        obs = load_observation(observation)
        obs.plot_on_axes(ax)

    ax.legend()
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*value_bounds)

    fig.savefig(f"{output_path}/AGN_radio_luminosity_function.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(description="AGN Radio Luminosity Function")

    snapshot_filenames = [
        f"{directory}/{snapshot}"
        for directory, snapshot in zip(
            arguments.directory_list, arguments.snapshot_list
        )
    ]

    plt.style.use(arguments.stylesheet_location)

    obs_data = glob.glob(
        f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}/data/BlackHoleAGNRadioLuminosityFunction/*.hdf5"
    )

    make_single_image(
        filenames=snapshot_filenames,
        names=arguments.name_list,
        x_bounds=x_bounds,
        value_bounds=value_bounds,
        number_of_simulations=arguments.number_of_inputs,
        output_path=arguments.output_directory,
        observational_data=obs_data,
    )
