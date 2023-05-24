import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from glob import glob

from swiftsimio import load
import unyt

bin_edges = 0.5 + np.arange(27, 63)
bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])


def get_data(run_directory: str, snapshot_name: str):

    print(run_directory)

    # Load snapshot
    data = load(f"{run_directory}/{snapshot_name}")

    # Load file with time-steps
    timesteps_glob = glob(f"{run_directory}/timesteps*.txt")
    print(timesteps_glob[0])
    timesteps_filename = timesteps_glob[0]
    timesteps_data = np.genfromtxt(
        timesteps_filename,
        skip_footer=5,
        loose=True,
        invalid_raise=False,
        usecols=(3, 4, 5),
        dtype=[("Redshift", "f4"), ("Timestep", "f4"), ("time_bin", "i1")],
    )

    # Compute relation between time-step bins and time-steps
    from_internal_time_to_cgs = data.metadata.internal_code_units[
        "Unit time in cgs (U_t)"
    ][0]
    snp_redshift_arg = np.argmin(np.abs(timesteps_data["Redshift"] - data.metadata.z))
    dt_base = unyt.unyt_quantity(
        timesteps_data["Timestep"][snp_redshift_arg] * from_internal_time_to_cgs, "s"
    ).to("kyr")
    timebin_base = timesteps_data["time_bin"][snp_redshift_arg]

    bh_minimum_allowed_timestep = unyt.unyt_array(
        float(
            data.metadata.parameters["COLIBREAGN:minimum_timestep_yr"].decode("utf-8")
        ),
        "yr",
    ).to("kyr")
    bh_minimum_allowed_timebin = timebin_base + np.log2(
        bh_minimum_allowed_timestep / dt_base
    )

    bh_timebins = data.black_holes.time_bins.value

    try:
        bh_minimal_timebins = data.black_holes.minimal_time_bins.value
    except AttributeError:
        bh_minimal_timebins = np.zeros_like(data.black_holes.time_bins.value)

    bh_accr_limited_timebins = timebin_base + np.log2(
        data.black_holes.accretion_limited_time_steps / dt_base
    )

    return (
        bh_timebins,
        bh_minimal_timebins,
        bh_accr_limited_timebins,
        bh_minimum_allowed_timebin,
    )


def make_single_image(
    run_directories, snapshot_names, names, number_of_simulations, output_path
):

    fig, ax = plt.subplots()

    ax.set_xlabel("Time-bin [-]")
    ax.set_ylabel("Fraction of black holes")

    for color, (directory, snapshot_name, name) in enumerate(
        zip(run_directories, snapshot_names, names)
    ):
        bh_timebins, bh_minimal_timebins, bh_accr_limited_timebins, bh_minimum_allowed_timebin = get_data(
            directory, snapshot_name
        )

        bin_bh_timebins, _ = np.histogram(bh_timebins, bins=bin_edges, density=True)
        bin_bh_minimal_timebins, _ = np.histogram(
            bh_minimal_timebins, bins=bin_edges, density=True
        )
        bin_bh_accr_limited_timebins, _ = np.histogram(
            bh_accr_limited_timebins, bins=bin_edges, density=True
        )

        (line1,) = ax.step(
            bins + 0.5, bin_bh_timebins, label=name, color=f"C{color}", ls="solid"
        )
        (line2,) = ax.step(
            bins + 0.5, bin_bh_minimal_timebins, color=f"C{color}", dashes=(2, 1)
        )
        (line3,) = ax.step(
            bins + 0.5,
            bin_bh_accr_limited_timebins,
            color=f"C{color}",
            dashes=(0.5, 0.5),
        )
        ax.axvline(bh_minimum_allowed_timebin, color=f"C{color}", dashes=(3, 1, 1, 1))

    ax.axvline(x=56, color="black", dashes=(4, 2), lw=1, label="Age of the Universe")
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
        Line2D([0], [0], color="black", dashes=(1.5, 1.5)),
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

    fig.savefig(f"{output_path}/bh_timebin_hist.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(description="BH time-step bins histogram")

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
