import matplotlib.pyplot as plt
import numpy as np

from swiftsimio import load

from unyt import mh, cm, Gyr
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
import glob
from velociraptor.observations import load_observation
from pylab import rcParams

# Set the limits of the figure.
x_bounds = [1e5,3e10] 
value_bounds = [1e-8,1e-1]
bins=20
def_value = -1.

def get_data(filename):

    data = load(filename)

    masses = 1e10 * np.array(data.black_holes.subgrid_masses.value)
    values = masses
    print(np.amax(values))
    
    return values

def calculate_medians(filename, x_bounds, value_bounds, bins):
  
    values = get_data(filename)
    values = np.log10(values)
    
    snapshot = load(filename)
    boxsize = snapshot.metadata.boxsize.to("Mpc")
    box_volume = boxsize[0] * boxsize[1] * boxsize[2]
    
    x_bins = np.linspace(
        np.log10(x_bounds[0]), np.log10(x_bounds[1]), bins
    )
    #print(x_bins)
    bin_width = (np.log10(x_bounds[1]) - np.log10(x_bounds[0])) / bins

    mass_func = []
    mass_func_errors=[]
    for x in x_bins:
        print(values)
        #print(values>(x-bin_width*0.5))
        values_sliced = values[(values>(x-bin_width*0.5))&(values<(x+bin_width*0.5))]
        
        mass_func.append(np.size(values_sliced)/box_volume)
        mass_func_errors.append(np.sqrt(np.size(values_sliced))/box_volume)

    return x_bins, np.log10(mass_func), np.log10(np.array(mass_func)+np.array(mass_func_errors))-np.log10(np.array(mass_func))


def make_single_image(filenames, names, x_bounds, value_bounds, number_of_simulations, output_path, observational_data):

    fig, ax = plt.subplots()

    ax.set_xlabel("Black Hole Mass $\log_{10}M_{\\rm BH}$ $[{\\rm M}_\odot]$")
    ax.set_ylabel("Black Hole Mass Function $\log_{10}\Phi$ [${\\rm Mpc}^{-3}{\\rm dex}^{-1}$]")
    ax.set_xscale('log')
    ax.set_yscale('log')

    for filename, name in zip(filenames, names):
        x_bins, mass_function, mass_function_errors = calculate_medians(filename, x_bounds, value_bounds, bins)
        print(mass_function)
        mask = mass_function>-100
        fill_plot, = ax.plot(10**x_bins[mask],10**mass_function[mask], label=name)
        ax.fill_between(10**x_bins[mask], 10**(mass_function[mask]-mass_function_errors[mask]),10**(mass_function[mask]+mass_function_errors[mask]),alpha=0.2, facecolor=fill_plot.get_color())
        #scatter_plot = ax.scatter(masses_most_massive, values_most_massive, facecolor=fill_plot.get_color())
        #ax.scatter(masses_rest, values_rest, s = 1.5, edgecolors='none', marker='o', alpha=0.5, facecolor=fill_plot.get_color())
        #ax.plot([1e-10,1e11],[0.01,0.01],color='black',linestyle=':',linewidth=0.75, label='$\dot{m}=0.01$')
    
    rcParams.update({"lines.markersize" : 5})
    for index, observation in enumerate(observational_data):
        obs = load_observation(observation)
        obs.plot_on_axes(ax)

    ax.legend()
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*value_bounds)

    fig.savefig(f"{output_path}/black_hole_mass_function.png")

    return


if __name__ == "__main__":
    from swiftpipeline.argumentparser import ScriptArgumentParser

    arguments = ScriptArgumentParser(description="Black Hole Mass Function")

    snapshot_filenames = [
        f"{directory}/{snapshot}"
        for directory, snapshot in zip(
            arguments.directory_list, arguments.snapshot_list
        )
    ]

    plt.style.use(arguments.stylesheet_location)
    
    obs_data = glob.glob(
    f"{arguments.config.config_directory}/{arguments.config.observational_data_directory}/data/BlackHoleMassFunction/*.hdf5"
    )

    make_single_image(
        filenames=snapshot_filenames,
        names=arguments.name_list,
        x_bounds=x_bounds,
        value_bounds=value_bounds,
        number_of_simulations=arguments.number_of_inputs,
        output_path=arguments.output_directory,
        observational_data=obs_data
    )
