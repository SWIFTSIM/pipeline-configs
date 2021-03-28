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


def register_spesific_star_formation_rates(self, catalogue):

    # Define aperture size in kpc
    aperture_sizes = [30, 100]

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


def register_star_metallicities(self, catalogue):

    # Define aperture size in kpc
    aperture_sizes = [30, 100]

    # Solar metallicity
    solar_metal_mass_fraction = 0.0126

    # Loop over apertures
    for aperture_size in aperture_sizes:

        try:
            metal_mass_fraction_star = (
                getattr(catalogue.apertures, f"zmet_star_{aperture_size}_kpc")
                / solar_metal_mass_fraction
            )
            metal_mass_fraction_star.name = (
                f"Star Metallicity $Z_*$ rel. to "
                f"$Z_\\odot={solar_metal_mass_fraction}$ ({aperture_size} kpc)"
            )
            setattr(
                self,
                f"star_metallicity_in_solar_{aperture_size}_kpc",
                metal_mass_fraction_star,
            )
        except AttributeError:
            pass

    return


def register_gas_phase_metallicities(self, catalogue):

    # Define aperture size in kpc
    aperture_sizes = [30, 100]

    # Metallicities relative to different units
    solar_metal_mass_fraction = 0.0126
    twelve_plus_log_OH_solar = 8.69
    minimal_twelve_plus_log_OH = 7.5

    # Loop over apertures
    for aperture_size in aperture_sizes:

        try:
            # Metallicity of star forming gas in units of solar metallicity
            metal_mass_fraction_gas = (
                getattr(catalogue.apertures, f"zmet_gas_sf_{aperture_size}_kpc")
                / solar_metal_mass_fraction
            )

            # Handle scenario where metallicity is zero, as we are bounded
            # by approx 1e-2 metal mass fraction anyway:
            metal_mass_fraction_gas[metal_mass_fraction_gas < 1e-5] = 1e-5

            # Get the log of the metallicity
            log_metal_mass_fraction_gas = np.log10(metal_mass_fraction_gas.value)

            # Get the O/H ratio
            twelve_plus_log_OH = unyt.unyt_array(
                twelve_plus_log_OH_solar + log_metal_mass_fraction_gas,
                units="dimensionless",
            )
            # Define name (label) of the derived field
            twelve_plus_log_OH.name = (
                f"Gas (SF) $12+\\log_{{10}}$O/H from "
                f"$Z$ (Solar={twelve_plus_log_OH_solar}) ({aperture_size} kpc)"
            )

            # Gas metallicity cannot be lower than the minimal value
            twelve_plus_log_OH[
                twelve_plus_log_OH < minimal_twelve_plus_log_OH
            ] = minimal_twelve_plus_log_OH

            setattr(
                self,
                f"gas_sf_twelve_plus_log_OH_{aperture_size}_kpc",
                twelve_plus_log_OH,
            )
        except AttributeError:
            pass

    return


def register_stellar_to_halo_mass_ratios(self, catalogue):

    # Define aperture size in kpc
    aperture_sizes = [30, 100]

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


def register_dust(self, catalogue):

    # Metal mass fractions of the gas
    metal_frac = catalogue.apertures.zmet_gas_100_kpc

    # Get total dust mass fractions by iterating through available dust types
    dust_fields = []
    for sub_path in dir(catalogue.dust_mass_fractions):
        if sub_path.startswith("dust_"):
            dust_fields.append(getattr(catalogue.dust_mass_fractions, sub_path).value)

    # We have found at least one dust field
    if len(dust_fields) > 0:
        # Sum along the zeroth axis of a 2D array
        total_dust_fraction = unyt.unyt_array(
            np.sum(dust_fields, axis=0), units="dimensionless"
        )
        dust_frac_error = ""

    # No dust fields have been found
    else:
        total_dust_fraction = unyt.unyt_array(
            np.zeros(metal_frac.size), units="dimensionless"
        )
        dust_frac_error = " (no dust field)"

    # Label for the dust-fraction derived field
    total_dust_fraction.name = f"$\\mathcal{{DTG}}${dust_frac_error}"

    # Compute total dust mass
    total_dust_mass = total_dust_fraction * catalogue.masses.mass_gas
    total_dust_mass.name = f"$M_{{\\rm dust}}${dust_frac_error}"

    # Compute to metal ratio
    dust_to_metals = total_dust_fraction / metal_frac
    dust_to_metals.name = f"$\\mathcal{{DTM}}${dust_frac_error}"

    # Compute dust to stellar ratio
    dust_to_stars = total_dust_mass / catalogue.apertures.mass_star_100_kpc
    dust_to_stars.name = f"$M_{{\\rm dust}}/M_*${dust_frac_error}"

    # Create mask for the galaxies where abundances are well defined
    setattr(self, "valid_abundances", np.isfinite(dust_to_metals))

    # Register derived fields with dust
    setattr(self, f"total_dust_masses_100_kpc", total_dust_mass)
    setattr(self, f"dust_to_metal_ratio_100_kpc", dust_to_metals)
    setattr(self, f"dust_to_gas_ratio_100_kpc", total_dust_fraction)
    setattr(self, f"dust_to_stellar_ratio_100_kpc", dust_to_stars)

    return


# Get depletion properties
def register_oxygen_to_hydrogen(self, catalogue):

    try:
        # Abundances measured over entire subhalo, using VR additional properties
        H_frac = getattr(catalogue.element_mass_fractions, "element_0")
        O_frac = getattr(catalogue.element_mass_fractions, "element_4")
        o_abundance = unyt.unyt_array(
            12 + np.log10(O_frac / (16 * H_frac)), "dimensionless"
        )
        o_abundance.name = "Gas $12+\\log_{10}({{\\rm O/H}})$"
        setattr(self, "gas_o_abundance", o_abundance)
    # In case the catalogue does not have oxygen
    except AttributeError:
        # We did not produce these quantities.
        setattr(
            self,
            "gas_o_abundance",
            unyt.unyt_array(
                unyt.unyt_array(
                    np.ones(np.size(catalogue.masses.mass_gas)), "dimensionless"
                ),
                name="$12+\\log_10({{\\rm O/H}})$ not found, default to 1",
            ),
        )

    return


def register_hi_masses(self, catalogue):

    # Define aperture size in kpc
    aperture_sizes = [30, 100]

    for aperture_size in aperture_sizes:

        # Fetch HI mass
        HI_mass = getattr(catalogue.gas_species_masses, f"HI_mass_{aperture_size}_kpc")

        # Label of the derived field
        HI_mass.name = f"$M_{{\\rm HI}}$ ({aperture_size} kpc)"

        # Register derived field
        setattr(self, f"gas_HI_mass_{aperture_size}_kpc", HI_mass)

    return


def register_dust_to_hi_ratio(self, catalogue):

    # Get total dust mass fractions by iterating through available dust types
    dust_fields = []
    for sub_path in dir(catalogue.dust_mass_fractions):
        if sub_path.startswith("dust_"):
            dust_fields.append(getattr(catalogue.dust_mass_fractions, sub_path).value)

    # We have found at least one dust field
    if len(dust_fields) > 0:
        # Sum along the zeroth axis of a 2D array
        total_dust_fraction = unyt.unyt_array(
            np.sum(dust_fields, axis=0), units="dimensionless"
        )
        total_dust_mass = total_dust_fraction * catalogue.masses.mass_gas

        # Fetch HI mass (in 100 kpc apertures)
        HI_mass_100_kpc = getattr(catalogue.gas_species_masses, "HI_mass_100_kpc")

        # Compute dust-to-HI mass ratios
        dust_to_hi_ratio = total_dust_mass / HI_mass_100_kpc

        # Register field
        dust_to_hi_ratio.name = "M_{\\rm dust}/M_{\\rm HI} (100 kpc)"
        setattr(self, "gas_dust_to_hi_ratio", dust_to_hi_ratio)

    # No dust fields have been found
    else:
        # We did not produce these quantities.
        setattr(
            self,
            "gas_dust_to_hi_ratio",
            unyt.unyt_array(
                unyt.unyt_array(
                    np.ones(np.size(catalogue.masses.mass_gas)), "dimensionless"
                ),
                name="$M_{\\rm dust}/M_{\\rm HI}$ not found, default to 1s",
            ),
        )

    return


def register_h2_masses(self, catalogue):

    # Define aperture size in kpc
    aperture_sizes = [30, 100]

    for aperture_size in aperture_sizes:

        # Fetch H2 mass
        H2_mass = getattr(catalogue.gas_species_masses, f"H2_mass_{aperture_size}_kpc")

        # Label of the derived field
        H2_mass.name = f"$M_{{\\rm H_2}}$ ({aperture_size} kpc)"

        # Register field
        setattr(self, f"gas_H2_mass_{aperture_size}_kpc", H2_mass)

    return


def register_cold_gas_mass_ratios(self, catalogue):

    # Define aperture size in kpc
    aperture_sizes = [30, 100]

    # Loop over aperture sizes
    for aperture_size in aperture_sizes:

        # Fetch HI and H2 masses in apertures of a given size
        HI_mass = getattr(catalogue.gas_species_masses, f"HI_mass_{aperture_size}_kpc")
        H2_mass = getattr(catalogue.gas_species_masses, f"H2_mass_{aperture_size}_kpc")

        # Compute neutral H mass (HI + H2)
        neutral_H_mass = HI_mass + H2_mass
        # Add label
        neutral_H_mass.name = f"$M_{{\\rm HI + H_2}}$ ({aperture_size} kpc)"

        # Fetch total stellar mass
        stellar_mass = getattr(catalogue.apertures, f"mass_star_{aperture_size}_kpc")
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
        neutral_H_to_baryonic_fraction.name = (
            f"$M_{{\\rm HI + H_2}}/((M_*+ M_{{\\rm HI + H_2}})$ ({aperture_size} kpc)"
        )

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
        neutral_H_to_sf_fraction[sf_mask] = neutral_H_mass[sf_mask] / sf_mass[sf_mask]
        neutral_H_to_sf_fraction.name = (
            f"$M_{{\\rm HI + H_2}}/M_{{\\rm SF}}$ ({aperture_size} kpc)"
        )

        # Compute HI mass to SF gas mass ratio
        HI_to_sf_fraction = unyt.unyt_array(
            np.zeros_like(HI_mass), units="dimensionless"
        )
        HI_to_sf_fraction[sf_mask] = HI_mass[sf_mask] / sf_mass[sf_mask]
        HI_to_sf_fraction.name = f"$M_{{\\rm HI}}/M_{{\\rm SF}}$ ({aperture_size} kpc)"

        # Compute H2 mass to SF gas mass ratio
        H2_to_sf_fraction = unyt.unyt_array(
            np.zeros_like(H2_mass), units="dimensionless"
        )
        H2_to_sf_fraction[sf_mask] = H2_mass[sf_mask] / sf_mass[sf_mask]
        H2_to_sf_fraction.name = f"$M_{{\\rm H_2}}/M_{{\\rm SF}}$ ({aperture_size} kpc)"

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
        setattr(
            self,
            f"gas_neutral_H_mass_{aperture_size}_kpc",
            neutral_H_mass,
        )

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
        setattr(self, f"gas_HI_to_sf_fraction_{aperture_size}_kpc", HI_to_sf_fraction)
        setattr(self, f"gas_H2_to_sf_fraction_{aperture_size}_kpc", H2_to_sf_fraction)
        setattr(
            self,
            f"gas_sf_to_stellar_fraction_{aperture_size}_kpc",
            sf_to_stellar_fraction,
        )
        setattr(self, f"has_neutral_gas_{aperture_size}_kpc", neutral_H_mass > 0.0)

    return


def register_species_fractions(self, catalogue):

    # Compute galaxy area (pi r^2)
    gal_area = (
        2
        * np.pi
        * catalogue.projected_apertures.projected_1_rhalfmass_star_100_kpc ** 2
    )

    # Mass in the projected apertures
    mstar_100 = catalogue.projected_apertures.projected_1_mass_star_100_kpc

    # Selection functions for the xGASS and xCOLDGASS surveys, used for the H species
    # fraction comparison. Note these are identical mass selections, but are separated
    # to keep survey selections explicit and to allow more detailed selection criteria
    # to be added for each.

    self.xgass_galaxy_selection = np.logical_and(
        catalogue.apertures.mass_star_100_kpc
        > unyt.unyt_quantity(10 ** 9, "Solar_Mass"),
        catalogue.apertures.mass_star_100_kpc
        < unyt.unyt_quantity(10 ** (11.5), "Solar_Mass"),
    )
    self.xcoldgass_galaxy_selection = np.logical_and(
        catalogue.apertures.mass_star_100_kpc
        > unyt.unyt_quantity(10 ** 9, "Solar_Mass"),
        catalogue.apertures.mass_star_100_kpc
        < unyt.unyt_quantity(10 ** (11.5), "Solar_Mass"),
    )

    # Register stellar mass density
    mu_star_100_kpc = mstar_100 / gal_area
    mu_star_100_kpc.name = "$M_{*, 100 {\\rm kpc}} / \\pi R_{*, 100 {\\rm kpc}}^2$"

    # Atomic hydrogen mass (100 kpc apertures)
    HI_mass_100_kpc = getattr(catalogue.gas_species_masses, "HI_mass_100_kpc")
    HI_mass_100_kpc.name = "HI Mass (100 kpc)"

    # Molecular hydrogen mass (100 kpc apertures)
    H2_mass_100_kpc = getattr(catalogue.gas_species_masses, "H2_mass_100_kpc")
    H2_mass_100_kpc.name = "H$_2$ Mass (100 kpc)"

    # Atomic hydrogen to stellar mass in the 100 kpc aperture
    hi_to_stellar_mass_100_kpc = HI_mass_100_kpc / catalogue.apertures.mass_star_100_kpc
    hi_to_stellar_mass_100_kpc.name = "$M_{\\rm HI} / M_*$ (100 kpc)"

    # Molecular hydrogen mass to stellar mass in the 100 kpc aperture
    h2_to_stellar_mass_100_kpc = H2_mass_100_kpc / catalogue.apertures.mass_star_100_kpc
    h2_to_stellar_mass_100_kpc.name = "$M_{\\rm H_2} / M_*$ (100 kpc)"

    # Neutral H / stellar mass
    neutral_to_stellar_mass_100_kpc = (
        hi_to_stellar_mass_100_kpc + h2_to_stellar_mass_100_kpc
    )
    neutral_to_stellar_mass_100_kpc.name = "$M_{\\rm HI + H_2} / M_*$ (100 kpc)"

    # Register all derived fields
    setattr(self, "mu_star_100_kpc", mu_star_100_kpc)

    setattr(self, "hi_to_stellar_mass_100_kpc", hi_to_stellar_mass_100_kpc)
    setattr(self, "h2_to_stellar_mass_100_kpc", h2_to_stellar_mass_100_kpc)
    setattr(self, "neutral_to_stellar_mass_100_kpc", neutral_to_stellar_mass_100_kpc)

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


# Register derived fields
register_spesific_star_formation_rates(self, catalogue)
register_star_metallicities(self, catalogue)
register_gas_phase_metallicities(self, catalogue)
register_stellar_to_halo_mass_ratios(self, catalogue)
register_dust(self, catalogue)
register_oxygen_to_hydrogen(self, catalogue)
register_hi_masses(self, catalogue)
register_h2_masses(self, catalogue)
register_dust_to_hi_ratio(self, catalogue)
register_cold_gas_mass_ratios(self, catalogue)
register_species_fractions(self, catalogue)
register_stellar_birth_densities(self, catalogue)
