"""Plot the stellar mass profiles in bins of halo mass"""

import matplotlib.pyplot as plt
import numpy as np
import unyt
import h5py as h5

from swiftsimio import load
from swiftpipeline.argumentparser import ScriptArgumentParser

import crossref as cx
import velociraptor as vr

import os

# Set number of halo mass bins to use
num_horizontal = 3
num_vertical = 3
num_bins = 9

mbinedges_log = np.linspace(10.5, 12.75, num_bins+1)
mbinedges = 10.0**mbinedges_log

num_radbins = 30
radbin_range_log = [-1.0, 2.0]
dlogr = (radbin_range_log[1] - radbin_range_log[0]) / num_radbins
radbin_range_log_full = [radbin_range_log[0] - dlogr, radbin_range_log[1]]

# Parameters for y axis scaling
Xs = [5.0, 2.5, 1.0, 5.0, 2.0, 1.0, 2.5, 1.5, 1.0]
yrange = ((0.0, 1.2), (0.0, 15.8), (0.0, 69.5))

def main():
    """Main function to create profiles."""

    print("Parsing input arguments...")
    arguments = ScriptArgumentParser(description="Stellar mass profiles")
    plt.style.use(arguments.stylesheet_location)
    names = arguments.name_list
    
    # Set up the SWIFTsimIO snapshot instances
    snapshots = [
        load(f"{directory}/{snapshot}") for directory, snapshot in zip(
            arguments.directory_list, arguments.snapshot_list)
    ]

    # Get the (median) redshift of all snapshots
    snap_redshifts = np.zeros(len(snapshots))
    for iisnap, isnap in enumerate(snapshots):
        snap_redshifts[iisnap] = isnap.metadata.redshift
        
    # Set up the Velociraptor catalogue instances
    catalogues = [
        vr.load(f"{directory}/{catalogue}") for directory, catalogue in zip(
            arguments.directory_list, arguments.catalogue_list)
    ]

    catalogue_files = [
        f"{directory}/{catalogue}" for directory, catalogue in zip(
            arguments.directory_list, arguments.catalogue_list)
    ]
            
    # Set up plot axes
    print("Set up axes...")
    fig, axes = setup_axes(snapshots[0])

    # Start by plotting comparison profiles
    comp_dir = os.path.dirname(os.path.realpath(__file__)) + '/comparison_data/'
    comp_sets = {
        'EAGLE-Ref-L25': ('EAGLE-Ref-L25_stellar_profiles.hdf5', 'lightgrey'),
        #'EAGLE-NoAGN-L25': ('EAGLE-NoAGN-L25_stellar_profiles.hdf5', 'powderblue'),
        #'EAGLE-Ref-L100': ('EAGLE-Ref-L100_stellar_profiles.hdf5', 'dimgrey'),
        'TNG100-1': ('IllustrisTNG-100-1_stellar_profiles.hdf5', 'rosybrown'),
        #'TNG100-2': ('IllustrisTNG-100-2_stellar_profiles.hdf5', 'peachpuff'),
    }

    set_lines = []
    set_names = list(comp_sets.keys())
    set_labels = set_names[:]
    for iiset, set_name in enumerate(set_names):
        with h5.File(f'{comp_dir}/{comp_sets[set_name][0]}', 'r') as f:
            redshifts = f['Redshifts'][...]
            ind_best = np.argmin(np.abs(redshifts - np.median(snap_redshifts)))
            z_best = redshifts[ind_best]
            set_labels[iiset] = set_labels[iiset] + f' (z = {z_best:.1f})'
            grp_name = f'z_{z_best:.3f}'.replace('.', 'p')
            print(f"Using group '{grp_name}'...")
            prof = f[f'{grp_name}/Profiles'][...]
            edges = f[f'{grp_name}/Edges'][...]
            prof_r = (edges[:-1] + edges[1:])/2

        for ibin in range(num_bins): 
            set_line = axes[ibin].plot(
                prof_r, prof[ibin, :] * Xs[ibin], color=comp_sets[set_name][1],
                label=set_name, linestyle='--', linewidth=1)[0]
            if ibin == 0:
                set_lines.append(set_line)
        
    # Add profiles for individual snapshots
    sim_lines = []
    for isnap, (snapshot, catalogue, catalogue_file, sim_name) in enumerate(
            zip(snapshots, catalogues, catalogue_files, names)):

        process_snapshot(snapshot, catalogue, catalogue_file, axes, f'C{isnap}',
                         sim_lines, sim_name, isnap)

    # Add legends
    sim_legend = axes[0].legend(sim_lines, names, frameon=1)
    set_legend = axes[1].legend(set_lines, set_labels, loc=3, prop={'size': 5},
                                frameon=1)

    set_frame = set_legend.get_frame() 
    set_frame.set_facecolor('white')
    sim_frame = sim_legend.get_frame() 
    sim_frame.set_facecolor('white')

    fig.savefig(f"{arguments.output_directory}/stellar_mass_profiles.png")



    
def setup_axes(snapshot):
    """Set up the (empty) plotting axes."""

    fig = plt.figure(figsize=(6.0, 6.0), constrained_layout=False)

    ax = []
    for axy in range(num_horizontal-1, -1, -1):
        for axx in range(num_vertical):

            # Add plot axis with correct size and position within full plot
            xwidth = (0.85 - (num_horizontal - 1) * 0.01) / num_horizontal
            ywidth = (0.85 - (num_horizontal - 1) * 0.01) / num_horizontal
            xlow = 0.1 + xwidth * axx + 0.01 * axx
            ylow = 0.1 + ywidth * axy + 0.01 * axy
            ax_curr = fig.add_axes([xlow, ylow, xwidth, ywidth])

            # Ensure that only left/bottom plots have labels
            if axx > 0:
                ax_curr.yaxis.label.set_visible(False)
                ax_curr.yaxis.set_ticklabels([])
            if axy > 0:
                ax_curr.xaxis.label.set_visible(False)
                ax_curr.xaxis.set_ticklabels([])

            ax_curr.set_xlim((-1.1, 2.0))
            if axy == 2:
                ax_curr.set_ylim(yrange[0])
            elif axy == 1:
                ax_curr.set_ylim(yrange[1])
            elif axy == 0:
                ax_curr.set_ylim(yrange[2])

            ibin = len(ax)
            m200_low, m200_high = mbinedges_log[ibin], mbinedges_log[ibin+1]
            yceil = ax_curr.get_ylim()[1]
            yr = ax_curr.get_ylim()[1] - ax_curr.get_ylim()[0]
            ytop = yceil * 0.965
            ax_curr.text(-0.9, ytop,
                         'log $M_\mathrm{200c}$:' f'\n{m200_low:.2f} - {m200_high:.2f}',
                         va='top')
            ax_curr.text(-0.9, ytop - yr*0.15, r'[$X =$' f'{Xs[ibin]:.1f}]', va='top', fontsize=7, color='dimgrey')
            if ibin > 0:
                ax_curr.text(0.8, ytop,
                             'log $M_\mathrm{star,\,med}$:', va='top')
            
            logeps = np.log10(
                snapshot.metadata.gravity_scheme[
                    'Maximal physical baryon softening length  '
                    '[internal units]']
                * 1e3)[0]
            logpeps = np.log10(
                snapshot.metadata.gravity_scheme[
                    'Maximal physical baryon softening length '
                    '(Plummer equivalent) [internal units]']
                * 1e3)[0]

            plt.fill_between((-1.1, logeps), (0, 0), (yceil, yceil),
                             color='grey', alpha=0.15)
            plt.fill_between((-1.1, logpeps), (0, 0), (yceil, yceil),
                             color='grey', alpha=0.15)
            
            # Add newly created axis to full stack
            ax.append(ax_curr)                

    # For simplicity, convert list of axes to a numpy array
    ax = np.array(ax)
    
    # Set (common) x and y axis labels
    fig.text(0.45, 0.06, "Galactic radius [log $r$ / pkpc]",
             va='top', ha='center')
    fig.text(0.05, 0.5, r"Stellar mass $M_\star\,/\,\mathrm{dlog}r\,\times X$ "
             r"[$10^9$ M$_\odot$]",
             va='center', ha='right', rotation=90.0)
    
    return fig, ax


def process_snapshot(snapshot, catalogue, catalogue_file, axes, colour,
                     sim_lines, sim_name, iisnap):
    """Calculate and plot the profiles for one snapshot.

    Parameters
    ----------
    snapshot : SWIFTDataset
        The snapshot for which to plot the profiles.
    catalogue : VELOCIRAPTOR instance
        The VR catalogue to link to.
    axes : ndarray(matplotlib axes)
        The pre-setup plot axes
    """

    # Extract relevant star particle info and match to subhaloes
    ids = snapshot.stars.particle_ids.value
    aexp = 1 / (1 + catalogue.redshift)
    star_pos = snapshot.stars.coordinates.to('Mpc').value * aexp
    star_haloes = cx.find_vr_haloes(ids, catalogue_file)

    # Extract relevant properties from VR catalogue
    vr_mstar = catalogue.masses.m_star_30kpc.to('Msun').value
    vr_x = catalogue.positions.xcminpot.to('Mpc').value
    vr_y = catalogue.positions.ycminpot.to('Mpc').value
    vr_z = catalogue.positions.zcminpot.to('Mpc').value
    vr_pos = np.vstack((vr_x, vr_y, vr_z)).T
    vr_m200c = catalogue.masses.mass_200crit.to('Msun').value
    vr_type = catalogue.structure_type.structuretype
    
    # Calculate radii for each star [kpc], accounting for periodic wrapping
    boxsize = snapshot.metadata.boxsize.to('Mpc').value
    star_deltapos = (star_pos - vr_pos[star_haloes, :])
    for idim in range(3):
        ind_low = np.nonzero(star_deltapos[:, idim] < -boxsize[idim]/2)
        star_deltapos[ind_low, idim] += boxsize[idim]
        ind_high = np.nonzero(star_deltapos[:, idim] > boxsize[idim]/2)
        star_deltapos[ind_high, idim] -= boxsize[idim]
    star_radii = np.linalg.norm(star_deltapos, axis=1) * 1e3

    edges = np.linspace(*radbin_range_log_full, num=num_radbins+1)
    dr_bin = edges[1:] - edges[:-1]
    
    for ibin in range(num_bins):
        m200_low, m200_high = mbinedges[ibin], mbinedges[ibin+1]
        ind_bin = np.nonzero((star_haloes >= 0) &
                             (vr_type[star_haloes] == 10) &
                             (vr_m200c[star_haloes] >= m200_low) &
                             (vr_m200c[star_haloes] < m200_high))[0]
        if len(ind_bin) == 0: continue

        logradii_bin = np.log10(
            np.clip(star_radii[ind_bin], 10.0**(radbin_range_log[0] - dlogr/2),
                    None))
        mstar_bin = snapshot.stars.masses.to('Msun').value[ind_bin]
        halo_bin = star_haloes[ind_bin]
        unique_halo_bin = np.unique(halo_bin)
        ngal = len(unique_halo_bin)
        mstar_med = np.log10(np.median(vr_mstar[unique_halo_bin]))
        
        # Find median galaxy per bin
        rad_sorter = np.argsort(logradii_bin)

        edge_indices = np.searchsorted(logradii_bin, edges, sorter=rad_sorter)

        profile = np.zeros(num_radbins)
        for iradbin in range(num_radbins):
            subind_radbin = np.arange(
                edge_indices[iradbin], edge_indices[iradbin+1])
            if len(subind_radbin) == 0: continue
            ind_radbin = rad_sorter[subind_radbin]
            unique_haloes, uhalo_inds = np.unique(
                halo_bin[ind_radbin], return_inverse=True)
            bin_hist, halobin_edges = np.histogram(
                uhalo_inds, bins=np.arange(len(unique_haloes)),
                weights=mstar_bin[ind_radbin])
            profile[iradbin] = np.median(bin_hist) / dr_bin[iradbin] / 1e9

            # Need to account for galaxies that have zero stars in this bin
            if len(bin_hist) > ngal:
                print("More haloes in bin than in total???")
                raise ValueError
            bin_hist_padded = np.zeros(ngal)
            bin_hist_padded[:len(bin_hist)] = bin_hist
            profile[iradbin] = np.median(bin_hist_padded)/dr_bin[iradbin]/1e9
            
        sim_line = axes[ibin].plot(edges[:-1] + dr_bin/2, profile * Xs[ibin],
                                   color=colour, label=sim_name)[0]

        if ibin == 0:
            # Only add each simulation once to the line list.
            sim_lines.append(sim_line)
        else:
            # Write out median stellar masses, but not in first bin because
            # it would clash with the legend there.
            yceil = axes[ibin].get_ylim()[1]
            yr = axes[ibin].get_ylim()[1] - axes[ibin].get_ylim()[0]
            ytop = yceil * 0.88
            axes[ibin].text(1.8, ytop - iisnap*0.06 * yr, f'{mstar_med:.2f}',
                            va='top', color=colour, ha='right')
            
        ylim = axes[ibin].get_ylim()[1]

        
if __name__ == '__main__':
    print("Creating stellar profiles...")
    main()
    print("Done!")
