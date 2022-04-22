import swiftsimio as sw
import numpy as np
import matplotlib.pyplot as plt
from swiftsimio.visualisation.projection import project_gas
from velociraptor.observations import load_observation

from unyt import Mpc, g, cm, mp

from astropy.cosmology import z_at_value
import astropy.units as u

import glob


def plot_cddf(snapshot_filenames, names, output_path, observational_data):
    simulation_lines = []
    simulation_labels = []

    fig, ax = plt.subplots()

    for snapshot_filename, name in zip(snapshot_filenames, names):

        # load the data
        data = sw.load(snapshot_filename)
        z = data.metadata.redshift
        boxsize = data.metadata.boxsize[0]

        # compute an appropriate number of pixels to use
        num_pix = int(4000.0 * (boxsize / (12.5 * Mpc)))

        # create the HI mass fraction dataset
        data.gas.HI_mass_fraction = (
            data.gas.masses
            * data.gas.element_mass_fractions.hydrogen
            * data.gas.species_fractions.HI
        )
        HI_map = project_gas(
            data,
            resolution=num_pix,
            project="HI_mass_fraction",
            parallel=True,
            backend="subsampled",
        )
        HI_numdens = HI_map.to(g * cm ** -2) / mp
        # convert from comoving to physical number densities
        HI_numdens *= (1.0 + z) ** 2

        # compute the histogram (density=True makes it a PDF)
        cddf, bin_edges = np.histogram(
            HI_numdens.flatten(), bins=np.logspace(12.0, 23.0, 100), density=True
        )

        # convert from d/dN to d/dlogN
        cddf *= (bin_edges[1:] - bin_edges[:-1]) / (
            np.log10(bin_edges[1:]) - np.log10(bin_edges[:-1])
        )

        # convert bin edges to bin centres
        bin_centres = 10.0 ** (
            0.5 * (np.log10(bin_edges[1:]) + np.log10(bin_edges[:-1]))
        )

        # figure out the box size in redshift space
        dist = data.metadata.cosmology.comoving_distance(z)
        dz = np.abs(
            z_at_value(
                data.metadata.cosmology.comoving_distance,
                dist + data.metadata.boxsize[0].to("Mpc").value * u.Mpc,
                zmin=z,
                zmax=z + 0.5,
            )
            - z
        )
        # correct the CDDF for line of sight sampling effects
        dX = (
            (data.metadata.cosmology.H0 / data.metadata.cosmology.H(z))
            * dz
            * (1 + z) ** 2
        )

        cddf /= dX.value

        simulation_lines.append(ax.loglog(bin_centres, cddf, lw=2)[0])

        simulation_labels.append(f"{name} ($z = {z:.1f}$)")

    for index, observation in enumerate(observational_data):
        obs = load_observation(observation)
        obs.plot_on_axes(ax)

    observation_legend = ax.legend(markerfirst=True, loc="lower left")

    ax.add_artist(observation_legend)

    simulation_legend = ax.legend(
        simulation_lines, simulation_labels, markerfirst=False, loc="upper right"
    )

    ax.set_xlabel("$N$(HI)")
    ax.set_ylabel("$\\log_{{10}} \\partial^2 n / \\partial \\log_{{10}} N \\partial X$")

    fig.savefig(f"{output_path}/cddf.png")


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(
        description="Column density distribution function plot."
    )

    snapshot_filenames = [
        f"{directory}/{snapshot}"
        for directory, snapshot in zip(
            arguments.directory_list, arguments.snapshot_list
        )
    ]

    plt.style.use(arguments.stylesheet_location)

    observational_data = glob.glob(
        f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}/data/ColumnDensityDistributionFunction/*.hdf5"
    )

    plot_cddf(
        snapshot_filenames,
        arguments.name_list,
        arguments.output_directory,
        observational_data,
    )
