"""
Makes a plot of the median temperature as a function of density for different
gas metallicity ranges.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl
import numpy as np
import swiftsimio as sw
import unyt
import scipy.stats as stats


# Set the limits of the figure.
density_bounds = [10 ** (-9.5), 1e7]  # in nh/cm^3
temperature_bounds = [10 ** (0), 10 ** (9.5)]  # in K
bins = 256
solar_metal_mass_fraction = 0.0134
# intervals in solar metallicity
Zbounds = [(0.1, 2.0), (0.01, 0.1), (0.001, 0.01), (0.0, 0.001)]

if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(description="Median temperature vs density plot.")

    snapshot_filenames = [
        f"{directory}/{snapshot}"
        for directory, snapshot in zip(
            arguments.directory_list, arguments.snapshot_list
        )
    ]

    pl.style.use(arguments.stylesheet_location)

    number_of_simulations = arguments.number_of_inputs
    sqrt_number_of_simulations = np.sqrt(number_of_simulations)
    horizontal_number = int(np.ceil(sqrt_number_of_simulations))
    # Ensure >= number_of_simulations plots in a grid
    vertical_number = int(np.ceil(number_of_simulations / horizontal_number))
    nlabel_per_plot = -1
    if vertical_number * horizontal_number == number_of_simulations:
        nlabel = len(Zbounds) + 1
        nlabel_per_plot = nlabel // number_of_simulations + (
            nlabel % number_of_simulations > 0
        )

    fig, ax = pl.subplots(
        vertical_number, horizontal_number, squeeze=True, sharex=True, sharey=True
    )

    ax = np.array([ax]) if number_of_simulations == 1 else ax

    if horizontal_number * vertical_number > number_of_simulations:
        for axis in ax.flat[number_of_simulations:]:
            axis.axis("off")

    # Set all valid on bottom row to have the horizontal axis label.
    for axis in np.atleast_2d(ax)[:][-1]:
        axis.set_xlabel("Density [$n_H$ cm$^{-3}$]")

    for axis in np.atleast_2d(ax).T[:][0]:
        axis.set_ylabel("Median Temperature [K]")

    ax = ax.flatten()

    nHbin = np.logspace(
        np.log10(density_bounds[0]), np.log10(density_bounds[1]), bins + 1
    )
    nHmid = 0.5 * (nHbin[1:] + nHbin[:-1])

    labels = [("C0", "all gas")]
    for iZ, (Zmin, Zmax) in enumerate(Zbounds):
        labels.append((f"C{iZ+1}", f"$Z \\in [{Zmin},{Zmax}]\\times Z_\\odot$"))

    for isnap, (snapshot, name) in enumerate(
        zip(snapshot_filenames, arguments.name_list)
    ):
        data = sw.load(snapshot)

        nH = (data.gas.densities.to_physical() / unyt.mh).to("cm**(-3)")
        T = data.gas.temperatures.to_physical().to("K")
        metal_mass_fraction = data.gas.metal_mass_fractions.to_physical().value

        with unyt.matplotlib_support:
            Tmed, _, _ = stats.binned_statistic(nH, T, statistic="median", bins=nHbin)
            label = labels[0][1] if (isnap == 0 and nlabel_per_plot >= 0) else None
            ax[isnap].loglog(nHmid, Tmed, label=label, color=labels[0][0])

            for iZ, (Zmin, Zmax) in enumerate(Zbounds):
                mask = (metal_mass_fraction >= Zmin * solar_metal_mass_fraction) & (
                    metal_mass_fraction < Zmax * solar_metal_mass_fraction
                )
                if mask.sum() > 0:
                    Tmed, _, _ = stats.binned_statistic(
                        nH[mask], T[mask], statistic="mean", bins=nHbin
                    )
                    label = None
                    if (
                        isnap * nlabel_per_plot <= iZ + 1
                        and (isnap + 1) * nlabel_per_plot > iZ + 1
                    ):
                        label = labels[iZ + 1][1]
                    ax[isnap].loglog(nHmid, Tmed, label=label, color=labels[iZ + 1][0])

            ax[isnap].text(
                0.025,
                0.975,
                name,
                ha="left",
                va="top",
                transform=ax[isnap].transAxes,
                fontsize=5,
                in_layout=False,
            )

    if nlabel_per_plot >= 0:
        for a in ax:
            a.legend(loc="best")
    else:
        handles = []
        for c, l in labels:
            handles.append(matplotlib.lines.Line2D([], [], color=c))
        ax[-1].legend(handles, [label for _, label in labels])
    pl.savefig(f"{arguments.output_directory}/median_temperature_density.png")
