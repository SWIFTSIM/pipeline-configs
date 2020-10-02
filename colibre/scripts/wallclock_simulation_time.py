"""
Plots wallclock v.s. simulation time.
"""

import unyt

import matplotlib.pyplot as plt
import numpy as np

from scipy import integrate
from swiftsimio import load

# cgs constants
constants = {"NEWTON_GRAVITY_CGS": 6.67408e-8,
             "SOLAR_MASS_IN_CGS":  1.98848e33,
             "PARSEC_IN_CGS":      3.08567758e18,
             "PROTON_MASS_IN_CGS": 1.672621898e-24,
             "BOLTZMANN_IN_CGS":   1.38064852e-16,
             "YEAR_IN_CGS":        3.15569252e7}

# Eagle cosmology
cosmology = {"Omega_m":      0.307,
             "Omega_lambda": 0.693,
             "h":            0.6777}

# Hubble parameter
H = lambda a: 100.0 * cosmology["h"] * np.sqrt(cosmology["Omega_m"] \
                    * np.power(a, -3.0) + cosmology["Omega_lambda"])

# Function to integrate to compute cosmic time
integrand = lambda a: 1. / H(a) / a

# Cosmic time in Gyr as a function of the scale factor
def cosmic_time(a): 
    return integrate.quad(integrand, 1e-12, a)[0] / 1e5 * constants["PARSEC_IN_CGS"] * 1e6 / \
                                                          (constants["YEAR_IN_CGS"] * 1e9)

# Function to display ticks in the second y-axis in the correct format
def tick_function(y):
    return ["%2.1f" % i for i in y]

try:
    plt.style.use("mnras.mplstyle")
except:
    pass

from glob import glob
from swiftpipeline.argumentparser import ScriptArgumentParser

arguments = ScriptArgumentParser(
    description="Creates a run performance plot: simulation time versus wall-clock time"
)

run_names = arguments.name_list
run_directories = [f"{directory}" for directory in arguments.directory_list]
snapshot_names = [f"{snapshot}" for snapshot in arguments.snapshot_list]
output_path = arguments.output_directory

plt.style.use(arguments.stylesheet_location)

fig, ax = plt.subplots()

sim_time_max = unyt.unyt_array(0, units="Gyr")

for run_name, run_directory, snapshot_name in zip(run_names,
                                                  run_directories,
                                                  snapshot_names):

    timesteps_glob = glob(f"{run_directory}/timesteps_*.txt")
    timesteps_filename = timesteps_glob[0]
    snapshot_filename = f"{run_directory}/{snapshot_name}"
    
    
    snapshot = load(snapshot_filename)
    data = np.genfromtxt(
        timesteps_filename, skip_footer=5, loose=True, 
        invalid_raise=False).T
     
    sim_time = unyt.unyt_array(data[1], units=snapshot.units.time).to("Gyr")
    wallclock_time = unyt.unyt_array(np.cumsum(data[-2]), units="ms").to("Hour")
   
    if sim_time[-1] > sim_time_max:
        sim_time_max = sim_time[-1]

    # Simulation data plotting
    (mpl_line,) = ax.plot(wallclock_time, sim_time, label=run_name)
    
    ax.scatter(wallclock_time[-1], sim_time[-1], color=mpl_line.get_color(),
               marker=".", zorder=10)

ax.set_xlim(0, None)
t_min, t_max = ax.set_ylim(0, sim_time_max * 1.05)

# Create second y-axis (to plot redshift alongsize cosmic time)
ax2 = ax.twinx()

# The ticks along the second axis
zticks=np.array([0, 0.2, 0.5, 1, 1.5, 2, 3, 4, 5, 10])
aticks=1./(1+zticks)

# To position the new ticks we need to know the corresponding cosmic times
cosmic_time_ticks = np.array([cosmic_time(a) for a in aticks])

# Attach the ticks to the new axis
ax2.set_yticks(cosmic_time_ticks)
ax2.set_yticklabels(tick_function(zticks))

ax.legend(loc="lower right")

ax.set_ylabel("Simulation time [Gyr]")
ax.set_xlabel("Wallclock time [Hours]")

ax2.set_ylabel("Redshift z")
ax2.set_ylim(0, t_max)

fig.savefig(f"{output_path}/wallclock_simulation_time.png")
