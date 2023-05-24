import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from glob import glob

from swiftsimio import load
import unyt

bin_edges = np.logspace(np.log10(50e-3), np.log10(50e7), 41)  # kyr
bins = 10 ** (0.5 * (np.log10(bin_edges[1:]) + np.log10(bin_edges[:-1])))


def get_data(run_directory: str, snapshot_name: str):

    # Load snapshot
    data = load(f"{run_directory}/{snapshot_name}")

    # Load file with time-steps
    timesteps_glob = glob(f"{run_directory}/timesteps*.txt")
    timesteps_filename = timesteps_glob[0]
    timesteps_data = np.genfromtxt(
        timesteps_filename,
        skip_footer=1,
        loose=True,
        invalid_raise=False,
        usecols=(3, 4, 5),
        dtype=[("Redshift", "f4"), ("Timestep", "f4"), ("time_bin", "i1")],
    )
    func_dt_z = interp1d(
        x=timesteps_data["Redshift"],
        y=timesteps_data["Timestep"] * 2.0 ** (-timesteps_data["time_bin"]),
        kind="linear",
        fill_value="extrapolate",
    )

    # Compute relation between time-step bins and time-steps
    from_internal_time_to_cgs = data.metadata.internal_code_units[
        "Unit time in cgs (U_t)"
    ][0]

    bh_timesteps = func_dt_z(data.metadata.z) * 2.0 ** data.black_holes.time_bins.value
    bh_timesteps = unyt.unyt_array(bh_timesteps * from_internal_time_to_cgs, "s").to(
        "kyr"
    )

    try:
        bh_minimal_timebin_z = (
            1.0 / data.black_holes.minimal_time_bin_scale_factors - 1.0
        )
        bh_minimal_timesteps = (
            func_dt_z(bh_minimal_timebin_z)
            * 2.0 ** data.black_holes.minimal_time_bins.value
        )
        bh_minimal_timesteps = unyt.unyt_array(
            bh_minimal_timesteps * from_internal_time_to_cgs, "s"
        ).to("kyr")
    except AttributeError:
        bh_minimal_timesteps = unyt.unyt_array(
            np.zeros_like(data.black_holes.time_bins.value), "kyr"
        )

    bh_accr_limited_timesteps = data.black_holes.accretion_limited_time_steps.to("kyr")
    bh_accr_minimum_allowed_timestep = unyt.unyt_array(
        float(
            data.metadata.parameters["COLIBREAGN:minimum_timestep_yr"].decode("utf-8")
        ),
        "yr",
    ).to("kyr")

    age_universe = data.metadata.cosmology.age(z=0).to("kyr").value

    return (
        bh_timesteps,
        bh_minimal_timesteps,
        bh_accr_limited_timesteps,
        bh_accr_minimum_allowed_timestep,
        age_universe,
    )


def make_single_image(
    run_directories, snapshot_names, names, number_of_simulations, output_path
):

    fig, ax = plt.subplots()
    ax.loglog()

    ax.set_xlabel("Time-step [kyr]")
    ax.set_ylabel("Fraction of black holes")

    for color, (directory, snapshot_name, name) in enumerate(
        zip(run_directories, snapshot_names, names)
    ):
        bh_timesteps, bh_minimal_timesteps, bh_accr_limited_timesteps, bh_accr_minimum_allowed_timestep, age_universe = get_data(
            directory, snapshot_name
        )

        bin_bh_timesteps, _ = np.histogram(bh_timesteps, bins=bin_edges, density=True)
        bin_bh_minimal_timesteps, _ = np.histogram(
            bh_minimal_timesteps, bins=bin_edges, density=True
        )
        bin_bh_accr_limited_timesteps, _ = np.histogram(
            bh_accr_limited_timesteps, bins=bin_edges, density=True
        )

        (line1,) = ax.plot(
            bins, bin_bh_timesteps, label=name, color=f"C{color}", ls="solid"
        )
        (line2,) = ax.plot(
            bins, bin_bh_minimal_timesteps, color=f"C{color}", dashes=(2, 1)
        )
        (line3,) = ax.plot(
            bins, bin_bh_accr_limited_timesteps, color=f"C{color}", dashes=(0.5, 0.5)
        )
        ax.axvline(
            bh_accr_minimum_allowed_timestep, color=f"C{color}", dashes=(3, 1, 1, 1)
        )

    ax.axvline(
        x=age_universe, color="black", dashes=(4, 2), lw=1, label="Age of the Universe"
    )
    ax.plot(
        [],
        [],
        color="black",
        dashes=(3, 1, 1, 1),
        lw=1,
        label="Minimum allowed time-step",
    )

    leg1 = ax.legend(loc="upper right")

    custom_lines = [
        Line2D([0], [0], color="black", linestyle="solid"),
        Line2D([0], [0], color="black", dashes=(2, 1)),
        Line2D([0], [0], color="black", dashes=(0.5, 0.5)),
    ]
    ax.legend(
        custom_lines,
        ["Current time-steps", "Minimal time-steps", "$\\dot{M}$-limited time-steps"],
        markerfirst=True,
        loc="center right",
        frameon=True,
    )
    ax.add_artist(leg1)

    fig.savefig(f"{output_path}/bh_timestep_hist.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(description="BH time-steps histogram")

    run_directories = [f"{directory}" for directory in arguments.directory_list]
    snapshot_names = [f"{snapshot}" for snapshot in arguments.snapshot_list]

    plt.style.use(arguments.stylesheet_location)

    make_single_image(
        run_directories=run_directories,
        snapshot_names=snapshot_names,
        names=arguments.name_list,
        number_of_simulations=arguments.number_of_inputs,
        output_path=arguments.output_directory,
    )
