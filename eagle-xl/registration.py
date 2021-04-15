"""
Registration of extra quantities for velociraptor catalogues (i.e. quantities
derived from the catalogue's internal properties).
This file calculates:
    + sSFR (30, 100 kpc) (specific_sfr_gas_{x}_kpc)
        This is the specific star formation rate of gas within those apertures.
        Only calculated for galaxies with a stellar mass greater than zero.
    + is_passive and is_active (30, 100 kpc) (is_passive_{x}_kpc)
        Boolean that determines whether or not a galaxy is passive. Marginal
        specific star formation rate is 1e-11 year^-1.
    + sfr_halo_mass (30, 100 kpc) (sfr_halo_mass_{x}_kpc)
        Star formation rate divided by halo mass with the star formation rate
        computed within apertures.
    + 12 + log(O/H) ({gas_sf_twelve_plus_log_OH_{x}_kpc, 30, 100 kpc)
        12 + log(O/H) based on metallicities. These should be removed at some point
        once velociraptor has a more sensible way of dealing with metallicity
        units.
    + metallicity_in_solar (star_metallicity_in_solar_{x}_kpc, 30, 100 kpc)
        Metallicity in solar units (relative to metal_mass_fraction).
    + stellar_mass_to_halo_mass_{x}_kpc for 30 and 100 kpc
        Stellar Mass / Halo Mass (both mass_200crit and mass_bn98) for 30 and 100 kpc
        apertures.
    + average of log of stellar birth densities (average_of_log_stellar_birth_density)
        velociraptor outputs the log of the quantity we need, so we take exp(...) of it
"""

# Define aperture size in kpc
aperture_sizes = [30, 100]

# Solar metal mass fraction used in Plöckinger S. & Schaye J. (2020)
solar_metal_mass_fraction = 0.0134

# Solar value for O/H
global twelve_plus_log_OH_solar
global minimal_twelve_plus_log_OH
twelve_plus_log_OH_solar = 8.69
minimal_twelve_plus_log_OH = 7.5

# Solar Fe abundance (from Wiersma et al 2009a)
solar_fe_abundance = 2.82e-5


def register_spesific_star_formation_rates(self, catalogue, aperture_sizes):

    # Lowest sSFR below which the galaxy is considered passive
    marginal_ssfr = unyt.unyt_quantity(1e-11, units=1 / unyt.year)

    # Loop over apertures
    for aperture_size in aperture_sizes:

        # Get the halo mass
        halo_mass = catalogue.masses.mass_200crit

        # Get stellar mass
        stellar_mass = getattr(catalogue.apertures, f"mass_star_{aperture_size}_kpc")
        # Need to mask out zeros, otherwise we get RuntimeWarnings
        good_stellar_mass = stellar_mass > unyt.unyt_quantity(0.0, stellar_mass.units)

        # Get star formation rate
        star_formation_rate = getattr(
            catalogue.apertures, f"sfr_gas_{aperture_size}_kpc"
        )

        # Compute specific star formation rate using the "good" stellar mass
        ssfr = unyt.unyt_array(np.zeros(len(star_formation_rate)), units=1 / unyt.year)
        ssfr[good_stellar_mass] = (
            star_formation_rate[good_stellar_mass] / stellar_mass[good_stellar_mass]
        )

        # Name (label) of the derived field
        ssfr.name = f"Specific SFR ({aperture_size} kpc)"

        # Mask for the passive objects
        is_passive = unyt.unyt_array(
            (ssfr < 1.01 * marginal_ssfr).astype(float), units="dimensionless"
        )
        is_passive.name = "Passive Fraction"

        # Mask for the active objects
        is_active = unyt.unyt_array(
            (ssfr > 1.01 * marginal_ssfr).astype(float), units="dimensionless"
        )
        is_active.name = "Active Fraction"

        # Get the specific star formation rate (per halo mass instead of stellar mass)
        sfr_M200 = star_formation_rate / halo_mass
        sfr_M200.name = "Star formation rate divided by halo mass"

        # Register derived fields with specific star formation rates
        setattr(self, f"specific_sfr_gas_{aperture_size}_kpc", ssfr)
        setattr(self, f"is_passive_{aperture_size}_kpc", is_passive)
        setattr(self, f"is_active_{aperture_size}_kpc", is_active)
        setattr(self, f"sfr_halo_mass_{aperture_size}_kpc", sfr_M200)

    return


def register_star_metallicities(self, catalogue, aperture_sizes, Z_sun):

    # Loop over apertures
    for aperture_size in aperture_sizes:

        try:
            metal_mass_fraction_star = (
                getattr(catalogue.apertures, f"zmet_star_{aperture_size}_kpc") / Z_sun
            )
            metal_mass_fraction_star.name = (
                f"Star Metallicity $Z_*$ rel. to "
                f"$Z_\\odot={Z_sun}$ ({aperture_size} kpc)"
            )
            setattr(
                self,
                f"star_metallicity_in_solar_{aperture_size}_kpc",
                metal_mass_fraction_star,
            )
        except AttributeError:
            pass

    return


def register_star_magnitudes(self, catalogue, aperture_sizes):

    bands = ["i", "g", "r", "H", "u", "J", "Y", "K", "z", "Z"]

    # Loop over apertures
    for aperture_size in aperture_sizes:
        for band in bands:
            try:

                L_AB = getattr(
                    catalogue.stellar_luminosities,
                    f"{band}_luminosity_{aperture_size}_kpc",
                )
                m_AB = np.copy(L_AB)
                mask = L_AB > 0.0
                m_AB[mask] = -2.5 * np.log10(m_AB[mask])
                m_AB = unyt.unyt_array(m_AB, units="dimensionless")
                m_AB.name = f"{band}-band magnitudes ({aperture_size} kpc)"
                setattr(self, f"magnitudes_{band}_band_{aperture_size}_kpc", m_AB)

            except AttributeError:
                pass

    return


def register_stellar_to_halo_mass_ratios(self, catalogue, aperture_sizes):

    # Loop over apertures
    for aperture_size in aperture_sizes:

        # Get the stellar mass in the aperture of a given size
        stellar_mass = getattr(catalogue.apertures, f"mass_star_{aperture_size}_kpc")

        # M200 critical
        halo_M200crit = catalogue.masses.mass_200crit
        smhm = stellar_mass / halo_M200crit
        smhm.name = f"$M_* / M_{{\\rm 200crit}}$ ({aperture_size} kpc)"
        setattr(self, f"stellar_mass_to_halo_mass_200crit_{aperture_size}_kpc", smhm)

        # BN98
        halo_MBN98 = catalogue.masses.mass_bn98
        smhm = stellar_mass / halo_MBN98
        smhm.name = f"$M_* / M_{{\\rm BN98}}$ ({aperture_size} kpc)"
        setattr(self, f"stellar_mass_to_halo_mass_bn98_{aperture_size}_kpc", smhm)

    return


def register_oxygen_to_hydrogen(
    self, catalogue, aperture_sizes, solar_metal_mass_fraction
):
    # Loop over apertures
    for aperture_size in aperture_sizes:
        metal_mass_fraction_gas = (
            getattr(catalogue.apertures, f"zmet_gas_sf_{aperture_size}_kpc")
            / solar_metal_mass_fraction
        )

        # Handle scenario where metallicity is zero, as we are bounded
        # by approx 1e-2 metal mass fraction anyway:
        metal_mass_fraction_gas[metal_mass_fraction_gas < 1e-5] = 1e-5

        log_metal_mass_fraction_gas = np.log10(metal_mass_fraction_gas.value)
        twelve_plus_log_OH = unyt.unyt_array(
            twelve_plus_log_OH_solar + log_metal_mass_fraction_gas,
            units="dimensionless",
        )
        twelve_plus_log_OH.name = f"Gas (SF) $12+\\log_{{10}}$O/H from $Z$ (Solar={twelve_plus_log_OH_solar}) ({aperture_size} kpc)"

        twelve_plus_log_OH[
            twelve_plus_log_OH < minimal_twelve_plus_log_OH
        ] = minimal_twelve_plus_log_OH

        setattr(
            self, f"gas_sf_twelve_plus_log_OH_{aperture_size}_kpc", twelve_plus_log_OH
        )
    return


def register_cold_gas_mass_ratios(self, catalogue, aperture_sizes):

    # Loop over aperture sizes
    for aperture_size in aperture_sizes:

        try:

            # Fetch HI and H2 masses in apertures of a given size
            HI_mass = getattr(
                catalogue.gas_hydrogen_species_masses, f"HI_mass_{aperture_size}_kpc"
            )
            H2_mass = getattr(
                catalogue.gas_hydrogen_species_masses, f"H2_mass_{aperture_size}_kpc"
            )

            # Compute neutral H mass (HI + H2)
            neutral_H_mass = HI_mass + H2_mass
            # Add label
            neutral_H_mass.name = f"$M_{{\\rm HI + H_2}}$ ({aperture_size} kpc)"

            # Fetch total stellar mass
            stellar_mass = getattr(
                catalogue.apertures, f"mass_star_{aperture_size}_kpc"
            )
            # Fetch mass of star-forming gas
            sf_mass = getattr(catalogue.apertures, f"mass_gas_sf_{aperture_size}_kpc")

            # Compute neutral H mass to stellar mass ratio
            neutral_H_to_stellar_fraction = neutral_H_mass / stellar_mass
            neutral_H_to_stellar_fraction.name = (
                f"$M_{{\\rm HI + H_2}} / M_*$ ({aperture_size} kpc)"
            )

            # Compute molecular H mass to molecular plus stellar mass fraction
            molecular_H_to_molecular_plus_stellar_fraction = H2_mass / (
                H2_mass + stellar_mass
            )
            molecular_H_to_molecular_plus_stellar_fraction.name = (
                f"$M_{{\\rm H_2}} / (M_* + M_{{\\rm H_2}})$ ({aperture_size} kpc)"
            )

            # Compute molecular H mass to neutral H mass ratio
            molecular_H_to_neutral_fraction = H2_mass / neutral_H_mass
            molecular_H_to_neutral_fraction.name = (
                f"$M_{{\\rm H_2}} / M_{{\\rm HI + H_2}}$ ({aperture_size} kpc)"
            )

            # Compute neutral H mass to baryonic mass fraction
            neutral_H_to_baryonic_fraction = neutral_H_mass / (
                neutral_H_mass + stellar_mass
            )
            neutral_H_to_baryonic_fraction.name = f"$M_{{\\rm HI + H_2}}/((M_*+ M_{{\\rm HI + H_2}})$ ({aperture_size} kpc)"

            # Compute HI mass to neutral H mass ratio
            HI_to_neutral_H_fraction = HI_mass / neutral_H_mass
            HI_to_neutral_H_fraction.name = (
                f"$M_{{\\rm HI}}/M_{{\\rm HI + H_2}}$ ({aperture_size} kpc)"
            )

            # Compute H2 mass to neutral H mass ratio
            H2_to_neutral_H_fraction = H2_mass / neutral_H_mass
            H2_to_neutral_H_fraction.name = (
                f"$M_{{\\rm H_2}}/M_{{\\rm HI + H_2}}$ ({aperture_size} kpc)"
            )

            # Compute SF mass to SF mass plus stellar mass ratio
            sf_to_sf_plus_stellar_fraction = unyt.unyt_array(
                np.zeros_like(neutral_H_mass), units="dimensionless"
            )
            # Select only good mass
            star_plus_sf_mass_mask = sf_mass + stellar_mass > 0.0 * sf_mass.units
            sf_to_sf_plus_stellar_fraction[star_plus_sf_mass_mask] = sf_mass[
                star_plus_sf_mass_mask
            ] / (sf_mass[star_plus_sf_mass_mask] + stellar_mass[star_plus_sf_mass_mask])
            sf_to_sf_plus_stellar_fraction.name = (
                f"$M_{{\\rm SF}}/(M_{{\\rm SF}} + M_*)$ ({aperture_size} kpc)"
            )

            # Select only the star-forming gas mass that is greater than zero
            sf_mask = sf_mass > 0.0 * sf_mass.units

            # Compute neutral H mass to SF gas mass ratio
            neutral_H_to_sf_fraction = unyt.unyt_array(
                np.zeros_like(neutral_H_mass), units="dimensionless"
            )
            neutral_H_to_sf_fraction[sf_mask] = (
                neutral_H_mass[sf_mask] / sf_mass[sf_mask]
            )
            neutral_H_to_sf_fraction.name = (
                f"$M_{{\\rm HI + H_2}}/M_{{\\rm SF}}$ ({aperture_size} kpc)"
            )

            # Compute HI mass to SF gas mass ratio
            HI_to_sf_fraction = unyt.unyt_array(
                np.zeros_like(HI_mass), units="dimensionless"
            )
            HI_to_sf_fraction[sf_mask] = HI_mass[sf_mask] / sf_mass[sf_mask]
            HI_to_sf_fraction.name = (
                f"$M_{{\\rm HI}}/M_{{\\rm SF}}$ ({aperture_size} kpc)"
            )

            # Compute H2 mass to SF gas mass ratio
            H2_to_sf_fraction = unyt.unyt_array(
                np.zeros_like(H2_mass), units="dimensionless"
            )
            H2_to_sf_fraction[sf_mask] = H2_mass[sf_mask] / sf_mass[sf_mask]
            H2_to_sf_fraction.name = (
                f"$M_{{\\rm H_2}}/M_{{\\rm SF}}$ ({aperture_size} kpc)"
            )

            # Compute SF gas mss to stellar mass ratio
            sf_to_stellar_fraction = unyt.unyt_array(
                np.zeros_like(neutral_H_mass), units="dimensionless"
            )
            # Select only good stellar mass
            m_star_mask = stellar_mass > 0.0 * stellar_mass.units
            sf_to_stellar_fraction[m_star_mask] = (
                sf_mass[m_star_mask] / stellar_mass[m_star_mask]
            )
            sf_to_stellar_fraction.name = f"$M_{{\\rm SF}}/M_*$ ({aperture_size} kpc)"

            # Finally, register all the above fields
            setattr(self, f"gas_neutral_H_mass_{aperture_size}_kpc", neutral_H_mass)

            setattr(
                self,
                f"gas_neutral_H_to_stellar_fraction_{aperture_size}_kpc",
                neutral_H_to_stellar_fraction,
            )
            setattr(
                self,
                f"gas_molecular_H_to_molecular_plus_stellar_fraction_{aperture_size}_kpc",
                molecular_H_to_molecular_plus_stellar_fraction,
            )
            setattr(
                self,
                f"gas_molecular_H_to_neutral_fraction_{aperture_size}_kpc",
                molecular_H_to_neutral_fraction,
            )
            setattr(
                self,
                f"gas_neutral_H_to_baryonic_fraction_{aperture_size}_kpc",
                neutral_H_to_baryonic_fraction,
            )
            setattr(
                self,
                f"gas_HI_to_neutral_H_fraction_{aperture_size}_kpc",
                HI_to_neutral_H_fraction,
            )
            setattr(
                self,
                f"gas_H2_to_neutral_H_fraction_{aperture_size}_kpc",
                H2_to_neutral_H_fraction,
            )
            setattr(
                self,
                f"gas_sf_to_sf_plus_stellar_fraction_{aperture_size}_kpc",
                sf_to_sf_plus_stellar_fraction,
            )
            setattr(
                self,
                f"gas_neutral_H_to_sf_fraction_{aperture_size}_kpc",
                neutral_H_to_sf_fraction,
            )
            setattr(
                self, f"gas_HI_to_sf_fraction_{aperture_size}_kpc", HI_to_sf_fraction
            )
            setattr(
                self, f"gas_H2_to_sf_fraction_{aperture_size}_kpc", H2_to_sf_fraction
            )
            setattr(
                self,
                f"gas_sf_to_stellar_fraction_{aperture_size}_kpc",
                sf_to_stellar_fraction,
            )
            setattr(self, f"has_neutral_gas_{aperture_size}_kpc", neutral_H_mass > 0.0)

        except AttributeError:
            pass

    return


def register_species_fractions(self, catalogue, aperture_sizes):

    # Loop over aperture sizes
    for aperture_size in aperture_sizes:

        try:

            # Compute galaxy area (pi r^2)
            gal_area = (
                2
                * np.pi
                * getattr(
                    catalogue.projected_apertures,
                    f"projected_1_rhalfmass_star_{aperture_size}_kpc",
                )
                ** 2
            )

            # Stellar mass
            M_star = getattr(catalogue.apertures, f"mass_star_{aperture_size}_kpc")
            M_star_projected = getattr(
                catalogue.projected_apertures,
                f"projected_1_mass_star_{aperture_size}_kpc",
            )

            # Selection functions for the xGASS and xCOLDGASS surveys, used for the H species
            # fraction comparison. Note these are identical mass selections, but are separated
            # to keep survey selections explicit and to allow more detailed selection criteria
            # to be added for each.

            self.xgass_galaxy_selection = np.logical_and(
                M_star > unyt.unyt_quantity(10 ** 9, "Solar_Mass"),
                M_star < unyt.unyt_quantity(10 ** (11.5), "Solar_Mass"),
            )
            self.xcoldgass_galaxy_selection = np.logical_and(
                M_star > unyt.unyt_quantity(10 ** 9, "Solar_Mass"),
                M_star < unyt.unyt_quantity(10 ** (11.5), "Solar_Mass"),
            )

            # Register stellar mass density in apertures
            mu_star = M_star_projected / gal_area
            mu_star.name = f"$M_{{*, {aperture_size} {{\\rm kpc}}}} / \\pi R_{{*, {aperture_size} {{\\rm kpc}}}}^2$"

            # Atomic hydrogen mass in apertures
            HI_mass = getattr(
                catalogue.gas_hydrogen_species_masses, f"HI_mass_{aperture_size}_kpc"
            )
            HI_mass.name = f"HI Mass ({aperture_size} kpc)"

            # Molecular hydrogen mass in apertures
            H2_mass = getattr(
                catalogue.gas_hydrogen_species_masses, f"H2_mass_{aperture_size}_kpc"
            )
            H2_mass.name = f"H$_2$ Mass ({aperture_size} kpc)"

            # Atomic hydrogen to stellar mass in apertures
            hi_to_stellar_mass = HI_mass / M_star
            hi_to_stellar_mass.name = f"$M_{{\\rm HI}} / M_*$ ({aperture_size} kpc)"

            # Molecular hydrogen mass to stellar mass in apertures
            h2_to_stellar_mass = H2_mass / M_star
            h2_to_stellar_mass.name = f"$M_{{\\rm H_2}} / M_*$ ({aperture_size} kpc)"

            # Neutral H / stellar mass
            neutral_to_stellar_mass = hi_to_stellar_mass + h2_to_stellar_mass
            neutral_to_stellar_mass.name = (
                f"$M_{{\\rm HI + H_2}} / M_*$ ({aperture_size} kpc)"
            )

            # Register all derived fields
            setattr(self, f"mu_star_{aperture_size}_kpc", mu_star)

            setattr(self, f"hi_to_stellar_mass_{aperture_size}_kpc", hi_to_stellar_mass)
            setattr(self, f"h2_to_stellar_mass_{aperture_size}_kpc", h2_to_stellar_mass)
            setattr(
                self,
                f"neutral_to_stellar_mass_{aperture_size}_kpc",
                neutral_to_stellar_mass,
            )

        except AttributeError:
            pass

    return


def register_stellar_birth_densities(self, catalogue):

    try:
        average_log_n_b = catalogue.stellar_birth_densities.logaverage
        stellar_mass = catalogue.apertures.mass_star_100_kpc

        exp_average_log_n_b = unyt.unyt_array(
            np.exp(average_log_n_b), units=average_log_n_b.units
        )

        # Ensure haloes with zero stellar mass have zero stellar birth densities
        no_stellar_mass = stellar_mass <= unyt.unyt_quantity(0.0, stellar_mass.units)
        exp_average_log_n_b[no_stellar_mass] = unyt.unyt_quantity(
            0.0, average_log_n_b.units
        )

        # Label of the derived field
        exp_average_log_n_b.name = "Stellar Birth Density (average of log)"

        # Register the derived field
        setattr(self, "average_of_log_stellar_birth_density", exp_average_log_n_b)

    # In case stellar birth densities are not present in the catalogue
    except AttributeError:
        pass

    return


def register_h2_masses(self, catalogue, aperture_sizes):

    # Loop over aperture sizes
    for aperture_size in aperture_sizes:

        try:

            # Fetch H2 mass
            H2_mass = getattr(
                catalogue.gas_hydrogen_species_masses, f"H2_mass_{aperture_size}_kpc"
            )

            # Label of the derived field
            H2_mass.name = f"$M_{{\\rm H_2}}$ ({aperture_size} kpc)"

            # Register field
            setattr(self, f"gas_H2_mass_{aperture_size}_kpc", H2_mass)

        except AttributeError:
            pass

    return


def register_hi_masses(self, catalogue, aperture_sizes):

    # Loop over aperture sizes
    for aperture_size in aperture_sizes:

        try:

            # Fetch HI mass
            HI_mass = getattr(
                catalogue.gas_hydrogen_species_masses, f"HI_mass_{aperture_size}_kpc"
            )

            # Label of the derived field
            HI_mass.name = f"$M_{{\\rm HI}}$ ({aperture_size} kpc)"

            # Register derived field
            setattr(self, f"gas_HI_mass_{aperture_size}_kpc", HI_mass)

        except AttributeError:
            pass

    return


def register_baryon_fractions(self, catalogue):
    try:
        Omega_m = catalogue.units.cosmology.Om0
        Omega_b = catalogue.units.cosmology.Ob0

        M_500 = catalogue.spherical_overdensities.mass_500_rhocrit
        M_500_gas = catalogue.spherical_overdensities.mass_gas_500_rhocrit
        M_500_star = catalogue.spherical_overdensities.mass_star_500_rhocrit
        M_500_baryon = M_500_gas + M_500_star

        f_b_500 = (M_500_baryon / M_500) / (Omega_b / Omega_m)
        name = "$f_{\\rm b, 500, true} / (\\Omega_{\\rm b} / \\Omega_{\\rm m})$"
        f_b_500.name = name

        f_gas_500 = (M_500_gas / M_500) / (Omega_b / Omega_m)
        name = "$f_{\\rm gas, 500, true} / (\\Omega_{\\rm b} / \\Omega_{\\rm m})$"
        f_gas_500.name = name

        setattr(self, "baryon_fraction_true_R500", f_b_500)
        setattr(self, "gas_fraction_true_R500", f_gas_500)
    except AttributeError:
        # We did not produce these quantities, let's make an array of ones.
        ones = unyt.unyt_array(
            np.ones(np.size(catalogue.masses.mass_200crit)), "dimensionless"
        )
        setattr(
            self,
            "baryon_fraction_true_R500",
            unyt.unyt_array(
                ones,
                name="$f_{\\rm b, 500, true} / (\\Omega_{\\rm b} / \\Omega_{\\rm m})$ not found, showing $1$",
            ),
        )
        setattr(
            self,
            "gas_fraction_true_R500",
            unyt.unyt_array(
                ones,
                name="$f_{\\rm gas, 500, true} / (\\Omega_{\\rm b} / \\Omega_{\\rm m})$ not found, showing $1$",
            ),
        )

    return


# Register derived fields
register_spesific_star_formation_rates(self, catalogue, aperture_sizes)
register_star_metallicities(self, catalogue, aperture_sizes, solar_metal_mass_fraction)
register_stellar_to_halo_mass_ratios(self, catalogue, aperture_sizes)
register_cold_gas_mass_ratios(self, catalogue, aperture_sizes)
register_oxygen_to_hydrogen(self, catalogue, aperture_sizes, solar_metal_mass_fraction)
register_species_fractions(self, catalogue, aperture_sizes)
register_stellar_birth_densities(self, catalogue)
register_hi_masses(self, catalogue, aperture_sizes)
register_h2_masses(self, catalogue, aperture_sizes)
register_baryon_fractions(self, catalogue)
register_star_magnitudes(self, catalogue, aperture_sizes)
