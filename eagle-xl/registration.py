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
        Stellar Mass / Halo Mass (mass_200crit) for 30 and 100 kpc apertures.
    + HI and H_2 masses (gas_HI_mass_Msun and gas_H2_mass_Msun).
    + baryon and gas fractions in R_(200,cr) normalized by the
        cosmic baryon fraction (baryon_fraction_true_R200, gas_fraction_true_R200).
"""

aperture_sizes = [30, 100]

# Specific star formation rate in apertures, as well as passive fraction
marginal_ssfr = unyt.unyt_quantity(1e-11, units=1 / unyt.year)

for aperture_size in aperture_sizes:
    halo_mass = catalogue.masses.mass_200crit

    stellar_mass = getattr(catalogue.apertures, f"mass_star_{aperture_size}_kpc")
    # Need to mask out zeros, otherwise we get RuntimeWarnings
    good_stellar_mass = stellar_mass > unyt.unyt_quantity(0.0, stellar_mass.units)

    star_formation_rate = getattr(catalogue.apertures, f"sfr_gas_{aperture_size}_kpc")

    ssfr = np.ones(len(star_formation_rate)) * marginal_ssfr
    ssfr[good_stellar_mass] = (
        star_formation_rate[good_stellar_mass] / stellar_mass[good_stellar_mass]
    )
    ssfr[ssfr < marginal_ssfr] = marginal_ssfr
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

# Now stellar mass - halo mass relation

for aperture_size in aperture_sizes:
    stellar_mass = getattr(catalogue.apertures, f"mass_star_{aperture_size}_kpc")
    halo_mass = catalogue.masses.mass_200crit

    smhm = stellar_mass / halo_mass
    name = f"$M_* / M_{{\\rm 200crit}}$ ({aperture_size} kpc)"
    smhm.name = name

    setattr(self, f"stellar_mass_to_halo_mass_{aperture_size}_kpc", smhm)

# Now HI masses

try:
    gas_mass = catalogue.masses.m_gas
    H_frac = getattr(catalogue.element_mass_fractions, "element_0")
    HI_frac = getattr(catalogue.species_fractions, "species_0")

    HI_mass = gas_mass * H_frac * HI_frac
    HI_mass.name = "$M_{HI}$"

    setattr(self, "gas_HI_mass_Msun", HI_mass)
except AttributeError:
    # We did not produce these quantities.
    setattr(
        self,
        "gas_HI_mass_Msun",
        unyt.unyt_array(
            catalogue.masses.m_gas, name="$M{\\rm HI}$ not found, showing $M_{\\rm g}$"
        ),
    )


# Now H2 masses

try:
    gas_mass = catalogue.masses.m_gas
    H_frac = getattr(catalogue.element_mass_fractions, "element_0")
    H2_frac = getattr(catalogue.species_fractions, "species_2")

    H2_mass = gas_mass * H_frac * H2_frac
    H2_mass.name = "$M_{H_2}$"

    setattr(self, "gas_H2_mass_Msun", H2_mass)
except AttributeError:
    # We did not produce these quantities.
    setattr(
        self,
        "gas_H2_mass_Msun",
        unyt.unyt_array(
            catalogue.masses.m_gas, name="$M{\\rm H_2}$ not found, showing $M_{\\rm g}$"
        ),
    )

# Now neutral H masses and fractions

try:
    gas_mass = catalogue.masses.m_gas
    H_frac = getattr(catalogue.element_mass_fractions, "element_0")
    HI_frac = getattr(catalogue.species_fractions, "species_0")
    H2_frac = getattr(catalogue.species_fractions, "species_2")

    HI_mass = gas_mass * H_frac * HI_frac
    H2_mass = gas_mass * H_frac * H2_frac
    neutral_H_mass = HI_mass + H2_mass
    neutral_H_mass.name = "$M_{HI + H_2}$"

    setattr(self, "gas_neutral_H_mass_Msun", neutral_H_mass)

    for aperture_size in aperture_sizes:
        stellar_mass = getattr(catalogue.apertures, f"mass_star_{aperture_size}_kpc")
        neutral_H_fraction = neutral_H_mass / stellar_mass
        neutral_H_fraction.name = "$M_{HI + H_2} / M_*$" + f"({aperture_size} kpc)"

        molecular_H_fraction = H2_mass / (H2_mass + stellar_mass)
        molecular_H_fraction.name = (
            "$M_{H_2} / (M_* + M_{H_2})$" + f"({aperture_size} kpc)"
        )

        setattr(self, f"gas_neutral_H_fraction_{aperture_size}_kpc", neutral_H_fraction)
        setattr(
            self, f"gas_molecular_H_fraction_{aperture_size}_kpc", molecular_H_fraction
        )

except AttributeError:
    # We did not produce these quantities.
    setattr(
        self,
        "gas_neutral_H_mass_Msun",
        unyt.unyt_array(
            catalogue.masses.m_gas,
            name="$M_{HI + H_2}$ not found, showing $M_{\\rm g}$",
        ),
    )
    # We did not produce these fractions, let's make an arrays of ones.
    ones = unyt.unyt_array(
        np.ones(np.size(catalogue.masses.mass_200crit)), "dimensionless"
    )
    for aperture_size in aperture_sizes:
        setattr(
            self,
            f"gas_neutral_H_fraction_{aperture_size}_kpc",
            unyt.unyt_array(
                catalogue.masses.m_gas,
                name="$M_{HI + H_2} / M_*$"
                + f"({aperture_size} kpc) not found, showing $1$",
            ),
        )
        setattr(
            self,
            f"gas_molecular_H_fraction_{aperture_size}_kpc",
            unyt.unyt_array(
                catalogue.masses.m_gas,
                name="$M_{H_2} / (M_* + M_{H_2})$"
                + f"({aperture_size} kpc) not found, showing $1$",
            ),
        )

# Now baryon fractions

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
