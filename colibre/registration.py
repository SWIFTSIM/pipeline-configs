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

# Aperture sizes in kpc in which stellar mass is computed
aperture_sizes = [30, 100]

# Specific star formation rate in apertures below which a galaxy is identified as
# passive
marginal_ssfr = unyt.unyt_quantity(1e-11, units=1 / unyt.year)

for aperture_size in aperture_sizes:
    halo_mass = catalogue.masses.mass_200crit

    stellar_mass = getattr(catalogue.apertures, f"mass_star_{aperture_size}_kpc")
    # Need to mask out zeros, otherwise we get RuntimeWarnings
    good_stellar_mass = stellar_mass > unyt.unyt_quantity(0.0, stellar_mass.units)

    star_formation_rate = getattr(catalogue.apertures, f"sfr_gas_{aperture_size}_kpc")

    ssfr = unyt.unyt_array(np.zeros(len(star_formation_rate)), units=1 / unyt.year)
    ssfr[good_stellar_mass] = (
        star_formation_rate[good_stellar_mass] / stellar_mass[good_stellar_mass]
    )
    ssfr.name = f"Specific SFR ({aperture_size} kpc)"

    is_passive = unyt.unyt_array(
        (ssfr < 1.01 * marginal_ssfr).astype(float), units="dimensionless"
    )
    is_passive.name = "Passive Fraction"

    is_active = unyt.unyt_array(
        (ssfr > 1.01 * marginal_ssfr).astype(float), units="dimensionless"
    )
    is_active.name = "Active Fraction"

    sfr_M200 = star_formation_rate / halo_mass
    sfr_M200.name = "Star formation rate divided by halo mass"

    setattr(self, f"specific_sfr_gas_{aperture_size}_kpc", ssfr)
    setattr(self, f"is_passive_{aperture_size}_kpc", is_passive)
    setattr(self, f"is_active_{aperture_size}_kpc", is_active)
    setattr(self, f"sfr_halo_mass_{aperture_size}_kpc", sfr_M200)

# Now metallicities relative to different units
solar_metal_mass_fraction = 0.0126
twelve_plus_log_OH_solar = 8.69
minimal_twelve_plus_log_OH = 7.5

for aperture_size in aperture_sizes:
    try:
        metal_mass_fraction_star = (
            getattr(catalogue.apertures, f"zmet_star_{aperture_size}_kpc")
            / solar_metal_mass_fraction
        )
        metal_mass_fraction_star.name = f"Star Metallicity $Z_*$ rel. to $Z_\\odot={solar_metal_mass_fraction}$ ({aperture_size} kpc)"
        setattr(
            self,
            f"star_metallicity_in_solar_{aperture_size}_kpc",
            metal_mass_fraction_star,
        )
    except AttributeError:
        pass

    try:
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
    except AttributeError:
        pass

metal_frac = catalogue.apertures.zmet_gas_100_kpc
nonmetal_frac = 1.0 - metal_frac

for aperture_size in aperture_sizes:
    stellar_mass = getattr(catalogue.apertures, f"mass_star_{aperture_size}_kpc")

    halo_M200crit = catalogue.masses.mass_200crit
    smhm = stellar_mass / halo_mass
    name = f"$M_* / M_{{\\rm 200crit}}$ ({aperture_size} kpc)"
    smhm.name = name
    setattr(self, f"stellar_mass_to_halo_mass_200crit_{aperture_size}_kpc", smhm)

    halo_MBN98 = catalogue.masses.mass_bn98
    smhm = stellar_mass / halo_MBN98
    name = f"$M_* / M_{{\\rm BN98}}$ ({aperture_size} kpc)"
    smhm.name = name
    setattr(self, f"stellar_mass_to_halo_mass_bn98_{aperture_size}_kpc", smhm)


# If present, iterate through available dust types
try:
    dust_fields = []
    for sub_path in dir(catalogue.dust_mass_fractions):
        if sub_path.startswith("dust_"):
            dust_fields.append(getattr(catalogue.dust_mass_fractions, sub_path))
    total_dust_fraction = sum(dust_fields)
    dust_frac_error = ""
except AttributeError:
    total_dust_fraction = np.zeros(stellar_mass.size)
    dust_frac_error = " (no dust field)"

total_dust_mass = total_dust_fraction * catalogue.masses.m_gas
name = f"$M_{{\\rm dust}}${dust_frac_error}"
total_dust_mass.name = name
total_dust_fraction.name = f"$\\mathcal{{DTG}}${dust_frac_error}"
total_dust_mass = total_dust_fraction * catalogue.masses.m_gas
total_dust_mass.name = f"$M_{{\\rm dust}}${dust_frac_error}"
dust_to_metals = total_dust_fraction / metal_frac
dust_to_metals.name = f"$\\mathcal{{DTM}}${dust_frac_error}"
dust_to_stars = total_dust_mass / catalogue.apertures.mass_star_100_kpc
dust_to_stars.name = f"$M_{{\\rm dust}}/M_*${dust_frac_error}"

nonmetal_frac = 1.0 - catalogue.apertures.zmet_gas_100_kpc

setattr(self, f"total_dust_masses_100_kpc", total_dust_mass)
setattr(self, f"dust_to_metal_ratio_100_kpc", dust_to_metals)
setattr(self, f"dust_to_gas_ratio_100_kpc", total_dust_fraction)
setattr(self, f"dust_to_stellar_ratio_100_kpc", dust_to_stars)

# Get depletion properties
try:
    # Abundances measured over entire subhalo, using VR additional properties
    gas_mass = catalogue.masses.m_gas
    H_frac = getattr(catalogue.element_mass_fractions, "element_0")
    O_frac = getattr(catalogue.element_mass_fractions, "element_4")
    o_abundance = unyt.unyt_array(
        12 + np.log10(O_frac / (16 * H_frac)), "dimensionless"
    )
    o_abundance.name = "Gas $12+\\log_{10}({{\\rm O/H}})$"
    setattr(self, "gas_o_abundance", o_abundance)

except AttributeError:
    # We did not produce these quantities.
    setattr(
        self,
        "gas_o_abundance",
        unyt.unyt_array(
            unyt.unyt_array(np.ones(np.size(catalogue.masses.m_gas)), "dimensionless"),
            name="$12+\\log_10({{\\rm O/H}})$ not found, default to 1s",
        ),
    )


# mask galaxies where abundances are well defined
self.valid_abundances = np.isfinite(dust_to_metals)

# Get HI masses
try:
    gas_mass = catalogue.masses.m_gas
    H_frac = getattr(catalogue.element_mass_fractions, "element_0")

    # Try CHIMES arrays
    if hasattr(catalogue.species_fractions, "species_7"):
        HI_frac = getattr(catalogue.species_fractions, "species_1")
    # If species_7 doesn't exist, switch to the (default) Table-cooling case
    else:
        HI_frac = getattr(catalogue.species_fractions, "species_0")

    HI_mass = gas_mass * H_frac * HI_frac
    HI_mass_wHe = gas_mass * nonmetal_frac * HI_frac
    HI_mass.name = "$M_{\\rm HI}$"

    setattr(self, "gas_HI_mass", HI_mass)
    setattr(self, "gas_HI_plus_He_mass", HI_mass_wHe)

except AttributeError:
    # We did not produce these quantities.
    setattr(
        self,
        "gas_HI_mass",
        unyt.unyt_array(
            catalogue.masses.m_gas, name="$M{\\rm HI}$ not found, showing $M_{\\rm g}$"
        ),
    )

# Get HI to dust ratio
try:
    dust_to_hi_ratio = total_dust_mass / HI_mass
    dust_to_hi_ratio.name = "M_{{\\rm dust}}/M_{{\\rm HI}}"
    setattr(self, "gas_dust_to_hi_ratio", dust_to_hi_ratio)

except AttributeError:
    # We did not produce these quantities.
    setattr(
        self,
        "gas_dust_to_hi_ratio",
        unyt.unyt_array(
            unyt.unyt_array(np.ones(np.size(catalogue.masses.m_gas)), "dimensionless"),
            name="$M_{{\\rm dust}}/M_{{\\rm HI}}$ not found, default to 1s",
        ),
    )


# Get H2 masses
try:
    gas_mass = catalogue.masses.m_gas
    H_frac = getattr(catalogue.element_mass_fractions, "element_0")

    # Try CHIMES arrays
    if hasattr(catalogue.species_fractions, "species_7"):
        H2_frac = getattr(catalogue.species_fractions, "species_7")
    # If species_7 doesn't exist, switch to the (default) Table-cooling case
    else:
        H2_frac = getattr(catalogue.species_fractions, "species_2")

    H2_mass = gas_mass * H_frac * H2_frac * 2.0
    H2_mass_wHe = gas_mass * nonmetal_frac * H2_frac * 2.0
    H2_mass.name = "$M_{\\rm H_2}$"

    setattr(self, "gas_H2_mass", H2_mass)
    setattr(self, "gas_H2_plus_He_mass", H2_mass_wHe)

except AttributeError:
    # We did not produce these quantities.
    setattr(
        self,
        "gas_H2_mass",
        unyt.unyt_array(
            catalogue.masses.m_gas, name="$M{\\rm H_2}$ not found, showing $M_{\\rm g}$"
        ),
    )

# Get neutral H masses and fractions
try:
    gas_mass = catalogue.masses.m_gas
    H_frac = getattr(catalogue.element_mass_fractions, "element_0")

    # Try CHIMES arrays
    if hasattr(catalogue.species_fractions, "species_7"):
        HI_frac = getattr(catalogue.species_fractions, "species_1")
        H2_frac = getattr(catalogue.species_fractions, "species_7")
    # If species_7 doesn't exist, switch to the (default) Table-cooling case
    else:
        HI_frac = getattr(catalogue.species_fractions, "species_0")
        H2_frac = getattr(catalogue.species_fractions, "species_2")

    HI_mass = gas_mass * H_frac * HI_frac
    H2_mass = gas_mass * H_frac * H2_frac * 2
    neutral_H_mass = HI_mass + H2_mass
    neutral_H_mass.name = "$M_{\\rm HI + H_2}$"

    setattr(self, "gas_neutral_H_mass", neutral_H_mass)

    for aperture_size in aperture_sizes:
        stellar_mass = getattr(catalogue.apertures, f"mass_star_{aperture_size}_kpc")
        sf_mass = getattr(catalogue.apertures, f"mass_gas_sf_{aperture_size}_kpc")
        neutral_H_to_stellar_fraction = neutral_H_mass / stellar_mass
        neutral_H_to_stellar_fraction.name = (
            f"$M_{{\\rm HI + H_2}} / M_*$ ({aperture_size} kpc)"
        )

        molecular_H_to_molecular_plus_stellar_fraction = H2_mass / (
            H2_mass + stellar_mass
        )
        molecular_H_to_molecular_plus_stellar_fraction.name = (
            f"$M_{{\\rm H_2}} / (M_* + M_{{\\rm H_2}})$ ({aperture_size} kpc)"
        )

        molecular_H_to_neutral_fraction = H2_mass / neutral_H_mass
        molecular_H_to_neutral_fraction.name = (
            f"$M_{{\\rm H_2}} / M_{{\\rm HI + H_2}}$ ({aperture_size} kpc)"
        )

        neutral_H_to_baryonic_fraction = neutral_H_mass / (
            neutral_H_mass + stellar_mass
        )
        neutral_H_to_baryonic_fraction.name = (
            f"$M_{{\\rm HI + H_2}}/((M_*+ M_{{\\rm HI + H_2}})$ ({aperture_size} kpc)"
        )

        HI_to_neutral_H_fraction = HI_mass / neutral_H_mass
        HI_to_neutral_H_fraction.name = (
            f"$M_{{\\rm HI}}/M_{{\\rm HI + H_2}}$ ({aperture_size} kpc)"
        )

        H2_to_neutral_H_fraction = H2_mass / neutral_H_mass
        H2_to_neutral_H_fraction.name = (
            f"$M_{{\\rm H_2}}/M_{{\\rm HI + H_2}}$ ({aperture_size} kpc)"
        )

        sf_to_sf_plus_stellar_fraction = sf_mass / (sf_mass + stellar_mass)
        sf_to_sf_plus_stellar_fraction.name = (
            f"$M_{{\\rm SF}}/(M_{{\\rm SF}} + M_*)$ ({aperture_size} kpc)"
        )

        mask = sf_mass > 0.0 * sf_mass.units

        neutral_H_to_sf_fraction = unyt.unyt_array(
            np.zeros_like(neutral_H_mass), units="dimensionless"
        )
        neutral_H_to_sf_fraction[mask] = neutral_H_mass[mask] / sf_mass[mask]
        neutral_H_to_sf_fraction.name = (
            f"$M_{{\\rm HI + H_2}}/M_{{\\rm SF}}$ ({aperture_size} kpc)"
        )

        HI_to_sf_fraction = unyt.unyt_array(
            np.zeros_like(HI_mass), units="dimensionless"
        )
        HI_to_sf_fraction[mask] = HI_mass[mask] / sf_mass[mask]
        HI_to_sf_fraction.name = f"$M_{{\\rm HI}}/M_{{\\rm SF}}$ ({aperture_size} kpc)"

        H2_to_sf_fraction = unyt.unyt_array(
            np.zeros_like(H2_mass), units="dimensionless"
        )
        H2_to_sf_fraction[mask] = H2_mass[mask] / sf_mass[mask]
        H2_to_sf_fraction.name = f"$M_{{\\rm H_2}}/M_{{\\rm SF}}$ ({aperture_size} kpc)"

        sf_to_stellar_fraction = sf_mass / stellar_mass
        sf_to_stellar_fraction.name = f"$M_{{\\rm SF}}/M_*$ ({aperture_size} kpc)"

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
            self, f"gas_HI_to_sf_fraction_{aperture_size}_kpc", HI_to_sf_fraction,
        )
        setattr(
            self, f"gas_H2_to_sf_fraction_{aperture_size}_kpc", H2_to_sf_fraction,
        )
        setattr(
            self,
            f"gas_sf_to_stellar_fraction_{aperture_size}_kpc",
            sf_to_stellar_fraction,
        )
        setattr(
            self, f"has_neutral_gas_{aperture_size}_kpc", neutral_H_mass > 0.0,
        )

except AttributeError:
    # We did not produce these quantities.
    setattr(
        self,
        "gas_neutral_H_mass",
        unyt.unyt_array(
            catalogue.masses.m_gas,
            name="$M_{\\rm HI + H_2}$ not found, showing $M_{\\rm g}$",
        ),
    )
    # We did not produce these fractions, let's make an arrays of ones.
    ones = unyt.unyt_array(
        np.ones(np.size(catalogue.masses.mass_200crit)), "dimensionless"
    )
    for aperture_size in aperture_sizes:
        setattr(
            self,
            f"gas_neutral_H_to_stellar_fraction_{aperture_size}_kpc",
            unyt.unyt_array(
                ones,
                name="$M_{{\\rm HI + H_2}} / M_*$ ({aperture_size} kpc) not found, showing $1$",
            ),
        )
        setattr(
            self,
            f"gas_molecular_H_to_molecular_plus_stellar_fraction_{aperture_size}_kpc",
            unyt.unyt_array(
                ones,
                name=f"$M_{{\\rm H_2}} / (M_* + M_{{\\rm H_2}})$ ({aperture_size} kpc) not found, showing $1$",
            ),
        )
        setattr(
            self,
            f"gas_molecular_H_to_neutral_fraction_{aperture_size}_kpc",
            unyt.unyt_array(
                ones,
                name=f"$M_{{\\rm H_2}} / M_{{\\rm HI + H_2}}$ ({aperture_size} kpc) not found, showing $1$",
            ),
        )
        setattr(
            self,
            f"gas_neutral_H_to_baryonic_fraction_{aperture_size}_kpc",
            unyt.unyt_array(ones, name="Fraction not found, showing $1$",),
        )
        setattr(
            self,
            f"gas_HI_to_neutral_H_fraction_{aperture_size}_kpc",
            unyt.unyt_array(ones, name="Fraction not found, showing $1$",),
        )
        setattr(
            self,
            f"gas_H2_to_neutral_H_fraction_{aperture_size}_kpc",
            unyt.unyt_array(ones, name="Fraction not found, showing $1$",),
        )
        setattr(
            self,
            f"gas_sf_to_sf_plus_stellar_fraction_{aperture_size}_kpc",
            unyt.unyt_array(ones, name="Fraction not found, showing $1$",),
        )
        setattr(
            self,
            f"gas_neutral_H_to_sf_fraction_{aperture_size}_kpc",
            unyt.unyt_array(ones, name="Fraction not found, showing $1$",),
        )
        setattr(
            self,
            f"gas_HI_to_sf_fraction_{aperture_size}_kpc",
            unyt.unyt_array(ones, name="Fraction not found, showing $1$",),
        )
        setattr(
            self,
            f"gas_H2_to_sf_fraction_{aperture_size}_kpc",
            unyt.unyt_array(ones, name="Fraction not found, showing $1$",),
        )
        setattr(
            self,
            f"gas_sf_to_stellar_fraction_{aperture_size}_kpc",
            unyt.unyt_array(ones, name="Fraction not found, showing $1$",),
        )
        setattr(
            self, f"has_neutral_gas_{aperture_size}_kpc", ones.astype(bool),
        )

# species fraction properties
gas_mass = catalogue.apertures.mass_gas_100_kpc
gal_area = (
    2 * np.pi * catalogue.projected_apertures.projected_1_rhalfmass_star_100_kpc ** 2
)
mstar_100 = catalogue.projected_apertures.projected_1_mass_star_100_kpc

# Selection functions for the xGASS and xCOLDGASS surveys, used for the H species fraction comparison.
# Note these are identical mass selections, but are separated to keep survey selections explicit
# and to allow more detailed selection criteria to be added for each.

self.xgass_galaxy_selection = np.logical_and(
    catalogue.apertures.mass_star_100_kpc > unyt.unyt_quantity(10 ** 9, "Solar_Mass"),
    catalogue.apertures.mass_star_100_kpc
    < unyt.unyt_quantity(10 ** (11.5), "Solar_Mass"),
)

self.xcoldgass_galaxy_selection = np.logical_and(
    catalogue.apertures.mass_star_100_kpc > unyt.unyt_quantity(10 ** 9, "Solar_Mass"),
    catalogue.apertures.mass_star_100_kpc
    < unyt.unyt_quantity(10 ** (11.5), "Solar_Mass"),
)

self.mu_star_100_kpc = mstar_100 / gal_area
self.mu_star_100_kpc.name = "$\\pi R_{*, 100 {\\rm kpc}}^2 / M_{*, 100 {\\rm kpc}}$"

try:
    H_frac = catalogue.element_mass_fractions.element_0
    hydrogen_frac_error = ""
except:

    H_frac = 0.0
    hydrogen_frac_error = "(no H abundance)"

try:
    # Test for CHIMES arrays
    species_HI = catalogue.species_fractions.species_1
    species_H2 = 2.0 * catalogue.species_fractions.species_7
    species_frac_error = ""
except:
    try:
        # Test for COLIBRE arrays
        species_HI = catalogue.species_fractions.species_0
        species_H2 = 2.0 * catalogue.species_fractions.species_2
        species_frac_error = ""
    except:
        species_HI = 0.0
        species_H2 = 0.0
        species_frac_error = "(no species field)"

total_error = f" {species_frac_error}{hydrogen_frac_error}"

self.neutral_hydrogen_mass_100_kpc = gas_mass * H_frac * species_HI
self.hi_to_stellar_mass_100_kpc = (
    self.neutral_hydrogen_mass_100_kpc / catalogue.apertures.mass_star_100_kpc
)
self.molecular_hydrogen_mass_100_kpc = gas_mass * H_frac * species_H2
self.h2_to_stellar_mass_100_kpc = (
    self.molecular_hydrogen_mass_100_kpc / catalogue.apertures.mass_star_100_kpc
)

self.h2_plus_he_to_stellar_mass_100_kpc = (
    H2_mass_wHe / catalogue.apertures.mass_star_100_kpc
)
self.hi_plus_he_to_stellar_mass_100_kpc = (
    HI_mass_wHe / catalogue.apertures.mass_star_100_kpc
)
self.cold_gas_to_stellar_mass_100_kpc = (
    HI_mass_wHe + H2_mass_wHe
) / catalogue.apertures.mass_star_100_kpc


self.neutral_to_stellar_mass_100_kpc = (
    self.hi_to_stellar_mass_100_kpc + self.h2_to_stellar_mass_100_kpc
)

self.neutral_hydrogen_mass_100_kpc.name = f"HI Mass (100 kpc){total_error}"
self.hi_to_stellar_mass_100_kpc.name = f"$M_{{\\rm HI}} / M_*$ (100 kpc) {total_error}"
self.molecular_hydrogen_mass_100_kpc.name = f"H$_2$ Mass (100 kpc){total_error}"
self.h2_to_stellar_mass_100_kpc.name = f"$M_{{\\rm H_2}} / M_*$ (100 kpc) {total_error}"
self.h2_plus_he_to_stellar_mass_100_kpc.name = (
    f"$M_{{\\rm H_2}} / M_*$ (100 kpc, inc. He) {total_error}"
)
self.hi_plus_he_to_stellar_mass_100_kpc.name = (
    f"$M_{{\\rm HI}} / M_*$ (100 kpc, inc. He) {total_error}"
)

self.neutral_to_stellar_mass_100_kpc.name = (
    f"$M_{{\\rm HI + H_2}} / M_*$ (100 kpc) {total_error}"
)

# Formatting script for average of the log of stellar birth densities
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

    name = "Stellar Birth Density (average of log)"
    exp_average_log_n_b.name = name
    setattr(self, "average_of_log_stellar_birth_density", exp_average_log_n_b)
except AttributeError:
    pass
