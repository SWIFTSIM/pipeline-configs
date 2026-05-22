"""
Registration of extra quantities for SOAP catalogues.
"""

aperture_sizes = [30, 50, 100]

solar_metal_mass_fraction = 0.0126
twelve_plus_log_OH_solar = 8.69
minimal_twelve_plus_log_OH = 7.5

# Band column index in stellar_luminosity (9 GAMA bands: u g r i z Y J H K)
BAND_COLUMNS = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "Z": 4, "Y": 5, "J": 6, "H": 7, "K": 8}

marginal_ssfr = unyt.unyt_quantity(1e-11, units=1 / unyt.year)

halo_mass = soap.spherical_overdensity_200_crit.total_mass
halo_mass_bn98 = soap.spherical_overdensity_bn98.total_mass

for aperture_size in aperture_sizes:
    sphere = getattr(soap, f"exclusive_sphere_{aperture_size}kpc")
    stellar_mass = sphere.stellar_mass
    star_formation_rate = sphere.star_formation_rate

    good_stellar_mass = stellar_mass > unyt.unyt_quantity(0.0, stellar_mass.units)

    ssfr = unyt.unyt_array(
        np.ones(len(star_formation_rate)) * marginal_ssfr.to(1 / unyt.year).value,
        units=1 / unyt.year,
    )
    ssfr[good_stellar_mass] = (
        star_formation_rate[good_stellar_mass] / stellar_mass[good_stellar_mass]
    ).to(1 / unyt.year)
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

    sphere.specific_star_formation_rate = ssfr
    sphere.is_passive = is_passive
    sphere.is_active = is_active
    sphere.star_formation_rate_per_halo_mass = sfr_M200

    # Stellar metallicity in solar units
    try:
        star_metallicity = sphere.stellar_mass_fraction_in_metals / solar_metal_mass_fraction
        star_metallicity.name = f"Star Metallicity $Z_*$ rel. to $Z_\\odot={solar_metal_mass_fraction}$ ({aperture_size} kpc)"
        sphere.stellar_metallicity_in_solar = star_metallicity
    except AttributeError:
        pass

    # Star-forming gas oxygen abundance from metallicity
    try:
        metal_frac_sf = sphere.star_forming_gas_mass_fraction_in_metals / solar_metal_mass_fraction
        metal_frac_sf[metal_frac_sf < 1e-5] = 1e-5
        twelve_plus_log_OH = unyt.unyt_array(
            twelve_plus_log_OH_solar + np.log10(metal_frac_sf.value),
            units="dimensionless",
        )
        twelve_plus_log_OH.name = f"Gas (SF) $12+\\log_{{10}}$O/H from $Z$ (Solar={twelve_plus_log_OH_solar}) ({aperture_size} kpc)"
        twelve_plus_log_OH[twelve_plus_log_OH < minimal_twelve_plus_log_OH] = minimal_twelve_plus_log_OH
        sphere.star_forming_gas_oxygen_abundance = twelve_plus_log_OH
    except AttributeError:
        pass

    # Stellar mass to halo mass ratios
    smhm_200 = stellar_mass / halo_mass
    smhm_200.name = f"$M_* / M_{{\\rm 200crit}}$ ({aperture_size} kpc)"
    sphere.stellar_mass_to_halo_mass_200crit = smhm_200

    smhm_bn98 = stellar_mass / halo_mass_bn98
    smhm_bn98.name = f"$M_* / M_{{\\rm BN98}}$ ({aperture_size} kpc)"
    sphere.stellar_mass_to_halo_mass_bn98 = smhm_bn98

    # Stellar magnitudes from luminosity
    try:
        lum = sphere.stellar_luminosity
        for band, col in BAND_COLUMNS.items():
            L_AB = lum[:, col]
            m_AB = np.copy(L_AB.value)
            mask = m_AB > 0.0
            m_AB[mask] = -2.5 * np.log10(m_AB[mask])
            mag = unyt.unyt_array(m_AB, units="dimensionless")
            mag.name = f"{band}-band AB magnitudes ({aperture_size} kpc)"
            setattr(sphere, f"stellar_magnitude_{band}_band", mag)
    except AttributeError:
        pass

    # Eddington-biased stellar mass (Behroozi 2019)
    bias_std = np.min(np.array([0.07 + 0.071 * float(soap.metadata.z), 0.3]))
    bias_factors = 10 ** (np.random.normal(0, bias_std, len(stellar_mass)))
    stellar_mass_with_bias = unyt.unyt_array(
        stellar_mass.value * bias_factors, units=stellar_mass.units
    )
    stellar_mass_with_bias.name = f"Stellar Mass $M_*$ ({aperture_size} kpc)"
    sphere.stellar_mass_with_eddington_bias = stellar_mass_with_bias

# Baryon fractions at R_500
Omega_m = soap.metadata.cosmology.Om0
Omega_b = soap.metadata.cosmology.Ob0

M_500 = soap.spherical_overdensity_500_crit.total_mass
M_500_gas = soap.spherical_overdensity_500_crit.gas_mass
M_500_star = soap.spherical_overdensity_500_crit.stellar_mass
M_500_baryon = M_500_gas + M_500_star

cosmic_baryon_fraction = Omega_b / Omega_m
so500 = soap.spherical_overdensity_500_crit

f_b = M_500_baryon / M_500 / cosmic_baryon_fraction
f_b.name = "$f_{\\rm b, 500, true} / (\\Omega_{\\rm b} / \\Omega_{\\rm m})$"
so500.baryon_fraction = f_b

f_gas = M_500_gas / M_500 / cosmic_baryon_fraction
f_gas.name = "$f_{\\rm gas, 500, true} / (\\Omega_{\\rm b} / \\Omega_{\\rm m})$"
so500.gas_fraction = f_gas

f_star = M_500_star / M_500 / cosmic_baryon_fraction
f_star.name = "$f_{\\rm star, 500, true} / (\\Omega_{\\rm b} / \\Omega_{\\rm m})$"
so500.stellar_fraction = f_star

# Stellar velocity dispersion from matrix diagonal (sigma^2 values)
for aperture_size in [10, 30]:
    sphere = getattr(soap, f"exclusive_sphere_{aperture_size}kpc")
    try:
        mat = sphere.stellar_velocity_dispersion_matrix
        sigma_3d = np.sqrt(mat[:, 0] + mat[:, 1] + mat[:, 2])
        sigma_3d.name = f"Stellar velocity dispersion ({aperture_size} kpc)"
        sphere.stellar_velocity_dispersion = sigma_3d
    except AttributeError:
        pass
