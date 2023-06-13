import swiftsimio as sw
import numpy as np
import matplotlib.pyplot as plt
from swiftsimio.visualisation.projection import project_gas
from velociraptor.observations import load_observation

import unyt

from astropy.cosmology import z_at_value
import astropy.units as u

import glob
import yaml


def plot_cddf(
    snapshot_filenames,
    names,
    output_path,
    observational_data,
    element,
    species,
    box_chunks=[1, 2, 1, 2],
    x_ranges=[(1.0e12, 1.0e23), (1.0e12, 1.0e23), (1.0e19, 1.0e23), (1.0e19, 1.0e23)],
    y_ranges=[(1.0e-5, 1.0e4), (1.0e-5, 1.0e4), (1.0e-5, 3.0e1), (1.0e-5, 3.0e1)],
    figure_names=[
        "column_density_distribution_function_chunk1_full_range",
        "column_density_distribution_function_chunk2_full_range",
        "column_density_distribution_function_chunk1_reduced_range",
        "column_density_distribution_function_chunk2_reduced_range",
    ],
    parallel=True,
):

    # set up bins in column density space. These are always the same.
    bin_edges = np.logspace(12.0, 23.0, 100)
    bin_centres = 10.0 ** (0.5 * (np.log10(bin_edges[1:]) + np.log10(bin_edges[:-1])))

    # determine the number of chunks to use to produce the maps
    # maps for smaller numbers of chunks are obtained by summing together maps
    max_N_chunk = np.max(box_chunks)

    # we (currently) only allow chunk numbers that are multiples of each other
    # there is no real need to do this, but it makes things a bit easier
    for N_chunk in np.unique(box_chunks):
        if not max_N_chunk % N_chunk == 0:
            raise AttributeError("Incompatible number of chunks found in list!")

    # we will store all the CDDF curves for all simulations in a large
    # dictionary
    # this way, we do not need to keep all the maps in memory until we make the
    # plot(s)
    cddf_data = {}
    for snapshot_filename, name in zip(snapshot_filenames, names):

        # the different CDDFs for different numbers of chunks are stored in a
        # dictionary as well
        cddf_data[snapshot_filename] = {}

        # load the data
        data = sw.load(snapshot_filename)
        z = data.metadata.redshift
        boxsize = data.metadata.boxsize[0]

        # compute an appropriate number of pixels to use
        num_pix = int(4000.0 * (boxsize / (12.5 * unyt.Mpc)))

        # generate the HI mass fraction dataset
        data.gas.species_mass_fraction = (
            data.gas.masses
            * getattr(data.gas.element_mass_fractions, element)
            * getattr(data.gas.species_fractions, species)
        )

        # create an empty array to store the different maps for all the chunks
        species_maps = np.zeros((max_N_chunk, num_pix, num_pix))
        for ichunk in range(max_N_chunk):
            # generate the map for this chunk and immediately convert from
            # mass column density to number column density
            species_map = (
                project_gas(
                    data,
                    resolution=num_pix,
                    project="species_mass_fraction",
                    parallel=parallel,
                    backend="subsampled",
                    region=[
                        0.0 * boxsize,
                        boxsize,
                        0.0 * boxsize,
                        boxsize,
                        (1.0 / max_N_chunk) * ichunk * boxsize,
                        (1.0 / max_N_chunk) * (ichunk + 1) * boxsize,
                    ],
                )
                / unyt.mp
            )
            # make sure the map has the right units:
            species_map = species_map.to("cm**(-2)")
            # (also taking into account the a^-2 cosmology factor)
            species_map *= (1.0 + z) ** 2
            # store the map in the array, this drops the units (which is fine)
            species_maps[ichunk] = species_map

        # now combine maps appropriately to generate the different CDDFs
        for N_chunk in np.unique(box_chunks):
            # convert the 3D array to a 4D array, where the first dimension
            # is the number of chunks we want and the second dimension is
            # automatically chosen by NumPy to accommodate this
            # then sum over this second dimension to convert back to a 3D array,
            # now containing the correct number of chunks.
            # finally flatten that array into a collection of individual sightlines
            # for which we can compute the CDDF
            species_map = (
                species_maps.reshape((N_chunk, -1, num_pix, num_pix))
                .sum(axis=1)
                .flatten()
            )

            # compute the histogram (density=True makes it a PDF)
            cddf, _ = np.histogram(species_map, bins=bin_edges, density=True)

            # convert from d/dN to d/dlogN
            cddf *= (bin_edges[1:] - bin_edges[:-1]) / (
                np.log10(bin_edges[1:]) - np.log10(bin_edges[:-1])
            )

            # figure out the box size in redshift space
            # note that this depends on the number of chunks, since more
            # chunks means a shorter z axis
            dist = data.metadata.cosmology.comoving_distance(z)
            dz = np.abs(
                z_at_value(
                    data.metadata.cosmology.comoving_distance,
                    dist + boxsize.to("Mpc").value / N_chunk * u.Mpc,
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

            # store the CDDF in the dictionary
            cddf_data[snapshot_filename][N_chunk] = cddf

    metadata = {}
    for snapshot_filename in snapshot_filenames:
        metadata[snapshot_filename] = {}

    # we have all the data, now create all the plots
    for N_chunk, x_range, y_range, figure_name in zip(
        box_chunks, x_ranges, y_ranges, figure_names
    ):

        simulation_lines = []
        simulation_labels = []

        fig, ax = plt.subplots()

        for snapshot_filename, name in zip(snapshot_filenames, names):

            simulation_lines.append(
                ax.loglog(bin_centres, cddf_data[snapshot_filename][N_chunk], lw=2)[0]
            )

            simulation_labels.append(f"{name} ($z = {z:.1f}$)")
            metadata[snapshot_filename][figure_name] = {
                "lines": {
                    "mean": {
                        "centers": bin_centres.tolist(),
                        "bins": bin_edges.tolist(),
                        "centers_units": "cm**(-2)",
                        "values": cddf_data[snapshot_filename][N_chunk].tolist(),
                        "values_units": "dimensionless",
                    }
                }
            }

        if species == "HI":
            for index, observation in enumerate(observational_data):
                obs = load_observation(observation)
                obs.plot_on_axes(ax)

            observation_legend = ax.legend(markerfirst=True, loc="lower left")
            ax.add_artist(observation_legend)

        simulation_legend = ax.legend(
            simulation_lines, simulation_labels, markerfirst=False, loc="upper right"
        )

        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)

        ax.set_xlabel(f"$N$({species})")
        ax.set_ylabel(
            "$\\log_{{10}} \\partial^2 n / \\partial \\log_{{10}} N \\partial X$"
        )

        fig.savefig(f"{output_path}/{figure_name}_{species}.png")

    for snapshot_filename in metadata:
        with open(f"{snapshot_filename.removesuffix('.hdf5')}_cddf.yml", "w") as handle:
            yaml.safe_dump(metadata[snapshot_filename], handle)


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(
        description="Column density distribution function plot.",
        additional_arguments={"parallel": True},
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
        element=arguments.element,
        species=arguments.species,
        parallel=arguments.parallel,
    )
