"""
Registration of extra quantities for SOAP catalogues.
"""

aperture_sizes = [30, 50, 100]

solar_metal_mass_fraction = 0.0134
twelve_plus_log_OH_solar = 8.69
solar_fe_abundance = 2.82e-5
stellar_mass_scatter_amplitude = 0.3


def register_sfr_derived(aperture_sizes):
    halo_mass = soap.spherical_overdensity_200_crit.total_mass
    marginal_ssfr = unyt.unyt_quantity(1e-11, units=1 / unyt.year)

    for aperture_size in aperture_sizes:
        sphere = getattr(soap, f"exclusive_sphere_{aperture_size}kpc")
        stellar_mass = sphere.stellar_mass
        sfr = sphere.star_formation_rate

        good_stellar_mass = stellar_mass > unyt.unyt_quantity(0.0, stellar_mass.units)
        ssfr = unyt.unyt_array(
            np.ones(len(sfr)) * marginal_ssfr.to(1 / unyt.year).value,
            units=1 / unyt.year,
        )
        ssfr[good_stellar_mass] = (
            sfr[good_stellar_mass] / stellar_mass[good_stellar_mass]
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

        sfr_per_halo_mass = sfr / halo_mass
        sfr_per_halo_mass.name = "Star formation rate divided by halo mass"

        sphere.specific_star_formation_rate = ssfr
        sphere.is_passive = is_passive
        sphere.is_active = is_active
        sphere.star_formation_rate_per_halo_mass = sfr_per_halo_mass


def register_stellar_metallicities(aperture_sizes, Z_sun):
    for aperture_size in aperture_sizes:
        sphere = getattr(soap, f"exclusive_sphere_{aperture_size}kpc")
        try:
            metallicity = sphere.stellar_mass_fraction_in_metals / Z_sun
            metallicity.name = (
                f"Star Metallicity $Z_*$ rel. to $Z_\\odot={Z_sun}$ ({aperture_size} kpc)"
            )
            sphere.stellar_metallicity_in_solar = metallicity
        except AttributeError:
            pass


def register_stellar_to_halo_mass_ratios(aperture_sizes):
    halo_mass_200crit = soap.spherical_overdensity_200_crit.total_mass
    halo_mass_bn98 = soap.spherical_overdensity_bn98.total_mass

    for aperture_size in aperture_sizes:
        sphere = getattr(soap, f"exclusive_sphere_{aperture_size}kpc")
        stellar_mass = sphere.stellar_mass

        smhm_200 = stellar_mass / halo_mass_200crit
        smhm_200.name = f"$M_* / M_{{\\rm 200crit}}$ ({aperture_size} kpc)"
        sphere.stellar_mass_to_halo_mass_200crit = smhm_200

        smhm_bn98 = stellar_mass / halo_mass_bn98
        smhm_bn98.name = f"$M_* / M_{{\\rm BN98}}$ ({aperture_size} kpc)"
        sphere.stellar_mass_to_halo_mass_bn98 = smhm_bn98


def register_oxygen_to_hydrogen(aperture_sizes):
    for aperture_size in aperture_sizes:
        sphere = getattr(soap, f"exclusive_sphere_{aperture_size}kpc")
        cold_dense_mass = sphere.gas_mass_in_cold_dense_gas
        mask = cold_dense_mass > unyt.unyt_quantity(0.0, cold_dense_mass.units)
        sphere.has_cold_dense_gas = mask

        # Linear mean O/H (diffuse)
        lin_O_H = sphere.linear_mass_weighted_diffuse_oxygen_over_hydrogen_of_gas
        O_H = np.zeros(len(cold_dense_mass))
        O_H[mask] = np.log10((lin_O_H[mask] / cold_dense_mass[mask]).value)
        O_abundance = unyt.unyt_array(12.0 + O_H, "dimensionless")
        O_abundance.name = f"SF Diffuse Gas $12+\\log_{{10}}({{\\rm O/H}})$ ({aperture_size} kpc)"
        sphere.gas_oxygen_abundance_linear_mean = O_abundance

        # Linear mean O/H (total = diffuse + dust)
        lin_O_H_total = sphere.linear_mass_weighted_oxygen_over_hydrogen_of_gas
        O_H = np.zeros(len(cold_dense_mass))
        O_H[mask] = np.log10((lin_O_H_total[mask] / cold_dense_mass[mask]).value)
        O_abundance = unyt.unyt_array(12.0 + O_H, "dimensionless")
        O_abundance.name = f"SF Total (Diffuse + Dust) Gas $12+\\log_{{10}}({{\\rm O/H}})$ ({aperture_size} kpc)"
        sphere.gas_oxygen_abundance_total_linear_mean = O_abundance

        # Log mean O/H (low floor, min = 1e-4 solar)
        log_O_H_low = sphere.logarithmic_mass_weighted_diffuse_oxygen_over_hydrogen_of_gas_low_limit
        O_H = np.zeros(len(cold_dense_mass))
        O_H[mask] = (log_O_H_low[mask] / cold_dense_mass[mask]).value
        O_abundance = unyt.unyt_array(12.0 + O_H, "dimensionless")
        O_abundance.name = f"SF Gas Diffuse $12+\\log_{{10}}({{\\rm O/H}})$ (Min = $10^{{-4}}$, {aperture_size} kpc)"
        sphere.gas_oxygen_abundance_log_mean_lowfloor = O_abundance

        # Log mean O/H (high floor, min = 1e-3 solar)
        log_O_H_high = sphere.logarithmic_mass_weighted_diffuse_oxygen_over_hydrogen_of_gas_high_limit
        O_H = np.zeros(len(cold_dense_mass))
        O_H[mask] = (log_O_H_high[mask] / cold_dense_mass[mask]).value
        O_abundance = unyt.unyt_array(12.0 + O_H, "dimensionless")
        O_abundance.name = f"SF Gas Diffuse $12+\\log_{{10}}({{\\rm O/H}})$ (Min = $10^{{-3}}$, {aperture_size} kpc)"
        sphere.gas_oxygen_abundance_log_mean_highfloor = O_abundance


def register_cold_dense_gas_metallicity(aperture_sizes, Z_sun, log_twelve_plus_logOH_solar):
    for aperture_size in aperture_sizes:
        sphere = getattr(soap, f"exclusive_sphere_{aperture_size}kpc")
        cold_dense_mass = sphere.gas_mass_in_cold_dense_gas
        cold_dense_metal_mass = sphere.gas_mass_in_cold_dense_diffuse_metals

        twelve_plus_logOH = np.zeros(len(cold_dense_mass)) + 1e-8
        mask = cold_dense_mass > unyt.unyt_quantity(0.0, cold_dense_mass.units)
        twelve_plus_logOH[mask] = (
            np.log10(
                (cold_dense_metal_mass[mask] / (Z_sun * cold_dense_mass[mask])).value
            )
            + log_twelve_plus_logOH_solar
        )

        O_abundance = unyt.unyt_array(twelve_plus_logOH, "dimensionless")
        O_abundance.name = (
            f"SF Gas $12+\\log_{{10}}({{\\rm O/H}})$ from $Z$ ({aperture_size} kpc)"
        )
        sphere.gas_oxygen_abundance_from_metallicity = O_abundance


def register_iron_to_hydrogen(aperture_sizes, fe_solar_abundance):
    for aperture_size in aperture_sizes:
        sphere = getattr(soap, f"exclusive_sphere_{aperture_size}kpc")
        stellar_mass = sphere.stellar_mass
        mask = stellar_mass > unyt.unyt_quantity(0.0, stellar_mass.units)

        # Linear mean Fe/H
        lin_Fe_H = sphere.linear_mass_weighted_iron_over_hydrogen_of_stars
        Fe_H = np.zeros(len(stellar_mass))
        Fe_H[mask] = (lin_Fe_H[mask] / stellar_mass[mask]).value
        Fe_abundance = unyt.unyt_array(Fe_H / fe_solar_abundance, "dimensionless")
        Fe_abundance.name = f"Stellar $10^{{\\rm [Fe/H]}}$ ({aperture_size} kpc)"
        sphere.stellar_iron_abundance_linear_mean = Fe_abundance

        # Log mean Fe/H (low floor)
        log_Fe_H_low = sphere.logarithmic_mass_weighted_iron_over_hydrogen_of_stars_low_limit
        Fe_H = np.zeros(len(stellar_mass))
        Fe_H[mask] = 10.0 ** (log_Fe_H_low[mask] / stellar_mass[mask]).value
        Fe_abundance = unyt.unyt_array(Fe_H / fe_solar_abundance, "dimensionless")
        Fe_abundance.name = (
            f"Stellar $10^{{\\rm [Fe/H]}}$ (Min = $10^{{-4}}$, {aperture_size} kpc)"
        )
        sphere.stellar_iron_abundance_log_mean_lowfloor = Fe_abundance

        # Log mean Fe/H (high floor)
        log_Fe_H_high = sphere.logarithmic_mass_weighted_iron_over_hydrogen_of_stars_high_limit
        Fe_H = np.zeros(len(stellar_mass))
        Fe_H[mask] = 10.0 ** (log_Fe_H_high[mask] / stellar_mass[mask]).value
        Fe_abundance = unyt.unyt_array(Fe_H / fe_solar_abundance, "dimensionless")
        Fe_abundance.name = (
            f"Stellar $10^{{\\rm [Fe/H]}}$ (Min = $10^{{-3}}$, {aperture_size} kpc)"
        )
        sphere.stellar_iron_abundance_log_mean_highfloor = Fe_abundance

        # Log mean SNIa Fe/H (low floor)
        try:
            log_Fe_snia_H = sphere.logarithmic_mass_weighted_iron_from_snia_over_hydrogen_of_stars_low_limit
            Fe_H = np.zeros(len(stellar_mass))
            Fe_H[mask] = 10.0 ** (log_Fe_snia_H[mask] / stellar_mass[mask]).value
            Fe_abundance = unyt.unyt_array(Fe_H / fe_solar_abundance, "dimensionless")
            Fe_abundance.name = (
                f"Stellar SNIa $10^{{\\rm [Fe/H]}}$ (Min = $10^{{-4}}$, {aperture_size} kpc)"
            )
            sphere.stellar_iron_from_snia_abundance_log_mean_lowfloor = Fe_abundance
        except AttributeError:
            pass


def register_star_mg_and_o_to_fe(aperture_sizes):
    # Solar mass ratios (Asplund et al. 2009)
    X_O_to_X_Fe_solar = 4.44
    X_Mg_to_X_Fe_solar = 0.55
    floor_value = -5.0

    for aperture_size in aperture_sizes:
        sphere = getattr(soap, f"exclusive_sphere_{aperture_size}kpc")
        frac_O = sphere.stellar_mass_fraction_in_oxygen
        frac_Mg = sphere.stellar_mass_fraction_in_magnesium
        frac_Fe = sphere.stellar_mass_fraction_in_iron

        mask_Mg = (frac_Fe > 0.0) & (frac_Mg > 0.0)
        mask_O = (frac_Fe > 0.0) & (frac_O > 0.0)

        Mg_over_Fe = floor_value * np.ones(len(frac_Fe))
        Mg_over_Fe[mask_Mg] = np.log10(
            (frac_Mg[mask_Mg] / frac_Fe[mask_Mg]).value
        ) - np.log10(X_Mg_to_X_Fe_solar)
        Mg_over_Fe = unyt.unyt_array(Mg_over_Fe, "dimensionless")
        Mg_over_Fe.name = f"[Mg/Fe]$_*$ ({aperture_size} kpc)"
        sphere.stellar_magnesium_over_iron = Mg_over_Fe

        O_over_Fe = floor_value * np.ones(len(frac_Fe))
        O_over_Fe[mask_O] = np.log10(
            (frac_O[mask_O] / frac_Fe[mask_O]).value
        ) - np.log10(X_O_to_X_Fe_solar)
        O_over_Fe = unyt.unyt_array(O_over_Fe, "dimensionless")
        O_over_Fe.name = f"[O/Fe]$_*$ ({aperture_size} kpc)"
        sphere.stellar_oxygen_over_iron = O_over_Fe


def register_dust(aperture_sizes):
    stellar_mass_100 = soap.exclusive_sphere_100kpc.stellar_mass

    for aperture_size in aperture_sizes:
        sphere = getattr(soap, f"exclusive_sphere_{aperture_size}kpc")
        gas_mass = sphere.gas_mass
        cold_dense_mass = sphere.gas_mass_in_cold_dense_gas
        cold_dense_metal_mass = sphere.gas_mass_in_cold_dense_diffuse_metals

        try:
            dust_total = sphere.dust_silicates_mass + sphere.dust_graphite_mass
        except AttributeError:
            dust_total = unyt.unyt_array(np.zeros(len(gas_mass)), units="Msun")

        dust_to_gas = dust_total / gas_mass
        dust_to_gas.name = f"$\\mathcal{{DTG}}$ ({aperture_size} kpc)"
        sphere.neutral_dust_to_gas_ratio = dust_to_gas

        dust_to_stars = dust_total / stellar_mass_100
        dust_to_stars.name = f"$M_{{\\rm dust}}/M_*$ ({aperture_size} kpc)"
        sphere.neutral_dust_to_stellar_ratio = dust_to_stars

        try:
            cold_dust = (
                sphere.dust_silicates_mass_in_cold_dense_gas
                + sphere.dust_graphite_mass_in_cold_dense_gas
            )
        except AttributeError:
            cold_dust = unyt.unyt_array(np.zeros(len(cold_dense_mass)), units="Msun")

        dtm_cold = unyt.unyt_array(np.zeros(len(cold_dense_mass)), "dimensionless")
        metal_mask = cold_dense_metal_mass > unyt.unyt_quantity(
            0.0, cold_dense_metal_mass.units
        )
        dtm_cold[metal_mask] = (
            cold_dust[metal_mask] / cold_dense_metal_mass[metal_mask]
        ).value
        dtm_cold.name = f"$\\mathcal{{DTM}}$ cold dense gas ({aperture_size} kpc)"
        sphere.cold_dense_dust_to_metal_ratio = dtm_cold


def register_neutral_gas_fractions(aperture_sizes):
    for aperture_size in aperture_sizes:
        sphere = getattr(soap, f"exclusive_sphere_{aperture_size}kpc")
        HI = sphere.atomic_hydrogen_mass
        H2 = sphere.molecular_hydrogen_mass
        neutral_H = HI + H2
        stellar_mass = sphere.stellar_mass
        sf_mass = sphere.star_forming_gas_mass

        neutral_to_stellar = neutral_H / stellar_mass
        neutral_to_stellar.name = f"$M_{{\\rm HI + H_2}} / M_*$ ({aperture_size} kpc)"
        sphere.neutral_hydrogen_to_stellar_mass_fraction = neutral_to_stellar

        H2_to_mol_plus_star = H2 / (H2 + stellar_mass)
        H2_to_mol_plus_star.name = (
            f"$M_{{\\rm H_2}} / (M_* + M_{{\\rm H_2}})$ ({aperture_size} kpc)"
        )
        sphere.molecular_hydrogen_to_molecular_plus_stellar_fraction = H2_to_mol_plus_star

        H2_to_neutral = H2 / neutral_H
        H2_to_neutral.name = f"$M_{{\\rm H_2}} / M_{{\\rm HI + H_2}}$ ({aperture_size} kpc)"
        sphere.molecular_hydrogen_to_neutral_fraction = H2_to_neutral

        neutral_to_baryonic = neutral_H / (neutral_H + stellar_mass)
        neutral_to_baryonic.name = (
            f"$M_{{\\rm HI + H_2}}/((M_*+ M_{{\\rm HI + H_2}})$ ({aperture_size} kpc)"
        )
        sphere.neutral_hydrogen_to_baryonic_fraction = neutral_to_baryonic

        HI_to_neutral = HI / neutral_H
        HI_to_neutral.name = f"$M_{{\\rm HI}}/M_{{\\rm HI + H_2}}$ ({aperture_size} kpc)"
        sphere.atomic_hydrogen_to_neutral_fraction = HI_to_neutral

        sf_mask = sf_mass > unyt.unyt_quantity(0.0, sf_mass.units)

        neutral_to_sf = unyt.unyt_array(np.zeros(len(neutral_H)), "dimensionless")
        neutral_to_sf[sf_mask] = (neutral_H[sf_mask] / sf_mass[sf_mask]).value
        neutral_to_sf.name = f"$M_{{\\rm HI + H_2}}/M_{{\\rm SF}}$ ({aperture_size} kpc)"
        sphere.neutral_hydrogen_to_star_forming_gas_fraction = neutral_to_sf

        HI_to_sf = unyt.unyt_array(np.zeros(len(HI)), "dimensionless")
        HI_to_sf[sf_mask] = (HI[sf_mask] / sf_mass[sf_mask]).value
        HI_to_sf.name = f"$M_{{\\rm HI}}/M_{{\\rm SF}}$ ({aperture_size} kpc)"
        sphere.atomic_hydrogen_to_star_forming_gas_fraction = HI_to_sf

        H2_to_sf = unyt.unyt_array(np.zeros(len(H2)), "dimensionless")
        H2_to_sf[sf_mask] = (H2[sf_mask] / sf_mass[sf_mask]).value
        H2_to_sf.name = f"$M_{{\\rm H_2}}/M_{{\\rm SF}}$ ({aperture_size} kpc)"
        sphere.molecular_hydrogen_to_star_forming_gas_fraction = H2_to_sf

        star_plus_sf_mask = (sf_mass + stellar_mass) > unyt.unyt_quantity(
            0.0, sf_mass.units
        )
        sf_to_sf_plus_star = unyt.unyt_array(np.zeros(len(neutral_H)), "dimensionless")
        sf_to_sf_plus_star[star_plus_sf_mask] = (
            sf_mass[star_plus_sf_mask]
            / (sf_mass[star_plus_sf_mask] + stellar_mass[star_plus_sf_mask])
        ).value
        sf_to_sf_plus_star.name = (
            f"$M_{{\\rm SF}}/(M_{{\\rm SF}} + M_*)$ ({aperture_size} kpc)"
        )
        sphere.star_forming_gas_to_sf_plus_stellar_fraction = sf_to_sf_plus_star

        m_star_mask = stellar_mass > unyt.unyt_quantity(0.0, stellar_mass.units)
        sf_to_star = unyt.unyt_array(np.zeros(len(neutral_H)), "dimensionless")
        sf_to_star[m_star_mask] = (sf_mass[m_star_mask] / stellar_mass[m_star_mask]).value
        sf_to_star.name = f"$M_{{\\rm SF}}/M_*$ ({aperture_size} kpc)"
        sphere.star_forming_gas_to_stellar_fraction = sf_to_star


def register_species_fractions(aperture_sizes):
    M_star_50 = soap.exclusive_sphere_50kpc.stellar_mass
    xgass_select = (M_star_50 > unyt.unyt_quantity(10**9, "Solar_Mass")) & (
        M_star_50 < unyt.unyt_quantity(10**11.5, "Solar_Mass")
    )
    soap.bound_subhalo.xgass_galaxy_selection = xgass_select
    soap.bound_subhalo.xcoldgass_galaxy_selection = xgass_select.copy()

    for aperture_size in aperture_sizes:
        sphere = getattr(soap, f"exclusive_sphere_{aperture_size}kpc")
        proj = getattr(soap, f"projected_aperture_{aperture_size}kpc_projx")

        M_star = sphere.stellar_mass
        M_star_proj = proj.stellar_mass
        r_half_proj = proj.half_mass_radius_stars

        gal_area = 2 * np.pi * r_half_proj**2
        mu_star = M_star_proj / gal_area
        mu_star.name = f"$M_{{*, {aperture_size} {{\\rm kpc}}}} / \\pi R_{{*, {aperture_size} {{\\rm kpc}}}}^2$"
        sphere.stellar_surface_mass_density = mu_star

        HI = sphere.atomic_hydrogen_mass
        H2 = sphere.molecular_hydrogen_mass
        He = sphere.helium_mass
        H = sphere.hydrogen_mass

        H2_with_He = H2 * (1.0 + He / H)
        H2_with_He.name = f"$M_{{\\rm H_2}}$ (incl. He, {aperture_size} kpc)"

        hi_to_star = HI / M_star
        hi_to_star.name = f"$M_{{\\rm HI}} / M_*$ ({aperture_size} kpc)"
        sphere.atomic_hydrogen_to_stellar_mass = hi_to_star

        h2_to_star = H2 / M_star
        h2_to_star.name = f"$M_{{\\rm H_2}} / M_*$ ({aperture_size} kpc)"
        sphere.molecular_hydrogen_to_stellar_mass = h2_to_star

        h2_plus_he_to_star = H2_with_He / M_star
        h2_plus_he_to_star.name = f"$M_{{\\rm H_2}} / M_*$ (incl. He, {aperture_size} kpc)"
        sphere.molecular_hydrogen_plus_helium_to_stellar_mass = h2_plus_he_to_star

        neutral_to_star = hi_to_star + h2_to_star
        neutral_to_star.name = f"$M_{{\\rm HI + H_2}} / M_*$ ({aperture_size} kpc)"
        sphere.neutral_hydrogen_to_stellar_mass = neutral_to_star

        jingle_select = (M_star > unyt.unyt_quantity(10**9, "Solar_Mass")) & (
            M_star < unyt.unyt_quantity(10**11, "Solar_Mass")
        )
        sphere.jingle_galaxy_selection = jingle_select


def register_stellar_birth_density():
    stellar_mass = soap.exclusive_sphere_100kpc.stellar_mass
    try:
        median_density = soap.bound_subhalo.median_stellar_birth_density
        density = unyt.unyt_array(median_density.value.copy(), units=median_density.units)
        no_stellar_mass = stellar_mass <= unyt.unyt_quantity(0.0, stellar_mass.units)
        density[no_stellar_mass] = unyt.unyt_quantity(0.0, density.units)
        density.name = "Stellar Birth Density (median)"
        soap.bound_subhalo.average_log_stellar_birth_density = density
    except AttributeError:
        pass


def register_gas_fraction():
    Omega_m = soap.metadata.cosmology.Om0
    Omega_b = soap.metadata.cosmology.Ob0
    cosmic_baryon_fraction = Omega_b / Omega_m

    so500 = soap.spherical_overdensity_500_crit
    M_500 = so500.total_mass
    M_500_gas = so500.gas_mass
    M_500_star = so500.stellar_mass
    M_500_baryon = M_500_gas + M_500_star

    f_b = M_500_baryon / M_500 / cosmic_baryon_fraction
    f_b.name = "$f_{\\rm b, 500, true} / (\\Omega_{\\rm b} / \\Omega_{\\rm m})$"
    so500.baryon_fraction = f_b

    f_gas = M_500_gas / M_500 / cosmic_baryon_fraction
    f_gas.name = "$f_{\\rm gas, 500, true} / (\\Omega_{\\rm b} / \\Omega_{\\rm m})$"
    so500.gas_fraction = f_gas

    f_star = M_500_star / M_500 / cosmic_baryon_fraction
    f_star.name = "$f_{\\rm star, 500, true} / (\\Omega_{\\rm b} / \\Omega_{\\rm m})$"
    so500.stellar_fraction = f_star


def register_los_stellar_velocity_dispersion():
    for aperture_size in [10, 30]:
        sphere = getattr(soap, f"exclusive_sphere_{aperture_size}kpc")
        try:
            mat = sphere.stellar_velocity_dispersion_matrix
            sigma_sq = (mat[:, 0] + mat[:, 1] + mat[:, 2]) / 3.0
            sigma_los = np.sqrt(sigma_sq)
            sigma_los.name = f"LOS stellar velocity dispersion ({aperture_size} kpc)"
            sphere.stellar_los_velocity_dispersion = sigma_los
        except AttributeError:
            pass


def register_stellar_mass_scatter(scatter_amplitude):
    sphere = soap.exclusive_sphere_50kpc
    stellar_mass = sphere.stellar_mass
    scattered = unyt.unyt_array(
        np.random.lognormal(
            np.log(stellar_mass.value), scatter_amplitude * np.log(10.0)
        ),
        units=stellar_mass.units,
    )
    scattered.name = (
        f"Stellar mass $M_*$ with {scatter_amplitude:.1f} dex scatter (50 kpc)"
    )
    sphere.stellar_mass_with_scatter = scattered


register_sfr_derived(aperture_sizes)
register_stellar_metallicities(aperture_sizes, solar_metal_mass_fraction)
register_stellar_to_halo_mass_ratios(aperture_sizes)
register_oxygen_to_hydrogen(aperture_sizes)
register_cold_dense_gas_metallicity(
    aperture_sizes, solar_metal_mass_fraction, twelve_plus_log_OH_solar
)
register_iron_to_hydrogen(aperture_sizes, solar_fe_abundance)
register_star_mg_and_o_to_fe(aperture_sizes)
register_dust(aperture_sizes)
register_neutral_gas_fractions(aperture_sizes)
register_species_fractions(aperture_sizes)
register_stellar_birth_density()
register_gas_fraction()
register_los_stellar_velocity_dispersion()
register_stellar_mass_scatter(stellar_mass_scatter_amplitude)
