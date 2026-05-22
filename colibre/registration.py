"""
Registration of extra quantities for SOAP catalogues.
"""

aperture_sizes = [30, 50, 100]

solar_metal_mass_fraction = 0.0134
twelve_plus_log_OH_solar = 8.69
solar_fe_abundance = 2.82e-5
solar_mg_abundance = 3.98e-5
stellar_mass_scatter_amplitude = 0.3

# Band column index in stellar_luminosity (9 GAMA bands: u g r i z Y J H K)
BAND_COLUMNS = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "Z": 4, "Y": 5, "J": 6, "H": 7, "K": 8}


def register_sfr_derived(aperture_sizes):
    halo_mass = soap.spherical_overdensity_200_crit.total_mass

    z = float(soap.metadata.z)
    if np.isclose(z, 0.0):
        marginal_ssfr = unyt.unyt_quantity(1e-11, units=1 / unyt.year)
    else:
        marginal_ssfr = unyt.unyt_quantity.from_astropy(
            0.2 * soap.metadata.cosmology.H(z)
        )

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

        halo_mask = halo_mass > unyt.unyt_quantity(0.0, halo_mass.units)
        sfr_per_halo_mass = unyt.unyt_array(
            np.zeros(len(sfr)), units=sfr.units / halo_mass.units
        )
        sfr_per_halo_mass[halo_mask] = (
            sfr[halo_mask] / halo_mass[halo_mask]
        )
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

        lin_O_H = sphere.linear_mass_weighted_diffuse_oxygen_over_hydrogen_of_gas
        O_H = np.zeros(len(cold_dense_mass))
        O_H[mask] = np.log10(lin_O_H[mask].to_physical_value("dimensionless"))
        O_abundance = unyt.unyt_array(12.0 + O_H, "dimensionless")
        O_abundance.name = f"SF Diffuse Gas $12+\\log_{{10}}({{\\rm O/H}})$ ({aperture_size} kpc)"
        sphere.gas_oxygen_abundance_linear_mean = O_abundance

        lin_O_H_total = sphere.linear_mass_weighted_oxygen_over_hydrogen_of_gas
        O_H = np.zeros(len(cold_dense_mass))
        O_H[mask] = np.log10(lin_O_H_total[mask].to_physical_value("dimensionless"))
        O_abundance = unyt.unyt_array(12.0 + O_H, "dimensionless")
        O_abundance.name = f"SF Total (Diffuse + Dust) Gas $12+\\log_{{10}}({{\\rm O/H}})$ ({aperture_size} kpc)"
        sphere.gas_oxygen_abundance_total_linear_mean = O_abundance

        log_O_H_low = sphere.logarithmic_mass_weighted_diffuse_oxygen_over_hydrogen_of_gas_low_limit
        O_H = np.zeros(len(cold_dense_mass))
        O_H[mask] = np.log10(log_O_H_low[mask].to_physical_value("dimensionless"))
        O_abundance = unyt.unyt_array(12.0 + O_H, "dimensionless")
        O_abundance.name = f"SF Gas Diffuse $12+\\log_{{10}}({{\\rm O/H}})$ (Min = $10^{{-4}}$, {aperture_size} kpc)"
        sphere.gas_oxygen_abundance_log_mean_lowfloor = O_abundance

        log_O_H_high = sphere.logarithmic_mass_weighted_diffuse_oxygen_over_hydrogen_of_gas_high_limit
        O_H = np.zeros(len(cold_dense_mass))
        O_H[mask] = np.log10(log_O_H_high[mask].to_physical_value("dimensionless"))
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

        lin_Fe_H = sphere.linear_mass_weighted_iron_over_hydrogen_of_stars
        Fe_H = np.zeros(len(stellar_mass))
        Fe_H[mask] = lin_Fe_H[mask].to_physical_value("dimensionless")
        Fe_abundance = unyt.unyt_array(Fe_H / fe_solar_abundance, "dimensionless")
        Fe_abundance.name = f"Stellar $10^{{\\rm [Fe/H]}}$ ({aperture_size} kpc)"
        sphere.stellar_iron_abundance_linear_mean = Fe_abundance

        log_Fe_H_low = sphere.logarithmic_mass_weighted_iron_over_hydrogen_of_stars_low_limit
        Fe_H = np.zeros(len(stellar_mass))
        Fe_H[mask] = log_Fe_H_low[mask].to_physical_value("dimensionless")
        Fe_abundance = unyt.unyt_array(Fe_H / fe_solar_abundance, "dimensionless")
        Fe_abundance.name = f"Stellar $10^{{\\rm [Fe/H]}}$ (Min = $10^{{-4}}$, {aperture_size} kpc)"
        sphere.stellar_iron_abundance_log_mean_lowfloor = Fe_abundance

        log_Fe_H_high = sphere.logarithmic_mass_weighted_iron_over_hydrogen_of_stars_high_limit
        Fe_H = np.zeros(len(stellar_mass))
        Fe_H[mask] = log_Fe_H_high[mask].to_physical_value("dimensionless")
        Fe_abundance = unyt.unyt_array(Fe_H / fe_solar_abundance, "dimensionless")
        Fe_abundance.name = f"Stellar $10^{{\\rm [Fe/H]}}$ (Min = $10^{{-3}}$, {aperture_size} kpc)"
        sphere.stellar_iron_abundance_log_mean_highfloor = Fe_abundance

        lin_Fe_snia_H = sphere.linear_mass_weighted_iron_from_snia_over_hydrogen_of_stars
        Fe_H = np.zeros(len(stellar_mass))
        Fe_H[mask] = lin_Fe_snia_H[mask].to_physical_value("dimensionless")
        Fe_abundance = unyt.unyt_array(Fe_H / fe_solar_abundance, "dimensionless")
        Fe_abundance.name = f"Stellar $10^{{\\rm [Fe_{{SNIa}}/H]}}$, {aperture_size} kpc)"
        sphere.stellar_iron_from_snia_abundance_linear_mean = Fe_abundance


def register_magnesium_to_hydrogen(aperture_sizes, mg_solar_abundance):
    for aperture_size in aperture_sizes:
        sphere = getattr(soap, f"exclusive_sphere_{aperture_size}kpc")
        stellar_mass = sphere.stellar_mass
        mask = stellar_mass > unyt.unyt_quantity(0.0, stellar_mass.units)

        try:
            lin_Mg_H = sphere.linear_mass_weighted_magnesium_over_hydrogen_of_stars
            Mg_H = np.zeros(len(stellar_mass))
            Mg_H[mask] = lin_Mg_H[mask].to_physical_value("dimensionless")
            Mg_abundance = unyt.unyt_array(Mg_H / mg_solar_abundance, "dimensionless")
            Mg_abundance.name = f"Stellar $10^{{\\rm [Mg/H]}}$ ({aperture_size} kpc)"
            sphere.stellar_magnesium_abundance_linear_mean = Mg_abundance

            log_Mg_H_low = sphere.logarithmic_mass_weighted_magnesium_over_hydrogen_of_stars_low_limit
            Mg_H = np.zeros(len(stellar_mass))
            Mg_H[mask] = log_Mg_H_low[mask].to_physical_value("dimensionless")
            Mg_abundance = unyt.unyt_array(Mg_H / mg_solar_abundance, "dimensionless")
            Mg_abundance.name = f"Stellar $10^{{\\rm [Mg/H]}}$ (Min = $10^{{-4}}$, {aperture_size} kpc)"
            sphere.stellar_magnesium_abundance_log_mean_lowfloor = Mg_abundance

            log_Mg_H_high = sphere.logarithmic_mass_weighted_magnesium_over_hydrogen_of_stars_high_limit
            Mg_H = np.zeros(len(stellar_mass))
            Mg_H[mask] = log_Mg_H_high[mask].to_physical_value("dimensionless")
            Mg_abundance = unyt.unyt_array(Mg_H / mg_solar_abundance, "dimensionless")
            Mg_abundance.name = f"Stellar $10^{{\\rm [Mg/H]}}$ (Min = $10^{{-3}}$, {aperture_size} kpc)"
            sphere.stellar_magnesium_abundance_log_mean_highfloor = Mg_abundance
        except AttributeError:
            pass


def register_star_mg_and_o_to_fe(aperture_sizes):
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
    for aperture_size in aperture_sizes:
        sphere = getattr(soap, f"exclusive_sphere_{aperture_size}kpc")
        gas_mass = sphere.gas_mass
        cold_dense_mass = sphere.gas_mass_in_cold_dense_gas
        cold_dense_metal_mass = sphere.gas_mass_in_cold_dense_diffuse_metals
        stellar_mass = sphere.stellar_mass

        try:
            dust_total = sphere.dust_silicates_mass + sphere.dust_graphite_mass
            neutral_dust = (
                sphere.dust_silicates_mass_in_atomic_gas
                + sphere.dust_graphite_mass_in_atomic_gas
                + sphere.dust_silicates_mass_in_molecular_gas
                + sphere.dust_graphite_mass_in_molecular_gas
            )
        except AttributeError:
            dust_total = unyt.unyt_array(np.zeros(len(gas_mass)), units="Msun")
            neutral_dust = unyt.unyt_array(np.zeros(len(gas_mass)), units="Msun")

        neutral_H = sphere.atomic_hydrogen_mass + sphere.molecular_hydrogen_mass
        dust_to_gas = neutral_dust / neutral_H
        dust_to_gas.name = f"$\\mathcal{{DTG}}$ ({aperture_size} kpc)"
        sphere.neutral_dust_to_gas_ratio = dust_to_gas

        dust_to_stars = dust_total / stellar_mass
        dust_to_stars.name = f"$M_{{\\rm dust}}/M_*$ ({aperture_size} kpc)"
        sphere.neutral_dust_to_stellar_ratio = dust_to_stars

        try:
            cold_dust = (
                sphere.dust_silicates_mass_in_cold_dense_gas
                + sphere.dust_graphite_mass_in_cold_dense_gas
            )
        except AttributeError:
            cold_dust = unyt.unyt_array(np.zeros(len(cold_dense_mass)), units="Msun")

        cold_dust_stored = unyt.unyt_array(cold_dust.value.copy(), units=cold_dust.units)
        cold_dust_stored.name = f"Cold Dense Dust Mass ({aperture_size} kpc)"
        sphere.cold_dense_dust_mass = cold_dust_stored

        try:
            mol_dust = (
                sphere.dust_silicates_mass_in_molecular_gas
                + sphere.dust_graphite_mass_in_molecular_gas
            )
        except AttributeError:
            mol_dust = unyt.unyt_array(np.zeros(len(gas_mass)), units="Msun")
        mol_dust.name = f"Molecular Dust Mass ({aperture_size} kpc)"
        sphere.molecular_dust_mass = mol_dust

        try:
            small_grain = sphere.dust_small_grain_mass
            large_grain = sphere.dust_large_grain_mass
            ratio = unyt.unyt_array(np.zeros(len(gas_mass)), "dimensionless")
            large_mask = large_grain > unyt.unyt_quantity(0.0, large_grain.units)
            ratio[large_mask] = (small_grain[large_mask] / large_grain[large_mask]).value
            ratio.name = f"Dust Small-to-Large Grain Ratio ({aperture_size} kpc)"
            sphere.dust_small_to_large_mass_ratio = ratio

            small_mol = sphere.dust_small_grain_mass_in_molecular_gas
            large_mol = sphere.dust_large_grain_mass_in_molecular_gas
            mol_ratio = unyt.unyt_array(np.zeros(len(gas_mass)), "dimensionless")
            large_mol_mask = large_mol > unyt.unyt_quantity(0.0, large_mol.units)
            mol_ratio[large_mol_mask] = (small_mol[large_mol_mask] / large_mol[large_mol_mask]).value
            mol_ratio.name = f"Molecular Dust Small-to-Large Grain Ratio ({aperture_size} kpc)"
            sphere.molecular_dust_small_to_large_mass_ratio = mol_ratio
        except AttributeError:
            pass

        O_H_solar = 10 ** (twelve_plus_log_OH_solar - 12)
        lin_O_H = sphere.linear_mass_weighted_oxygen_over_hydrogen_of_gas.to_physical_value("dimensionless")
        metal_frac_cd = solar_metal_mass_fraction * lin_O_H / O_H_solar
        dtm_cold = unyt.unyt_array(np.zeros(len(cold_dense_mass)), "dimensionless")
        metal_mask = (cold_dense_mass > unyt.unyt_quantity(0.0, cold_dense_mass.units)) & (metal_frac_cd > 0.0)
        dtm_cold[metal_mask] = (
            cold_dust[metal_mask] / (metal_frac_cd[metal_mask] * cold_dense_mass[metal_mask].to("Msun"))
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

        neutral_H_stored = unyt.unyt_array(neutral_H.value.copy(), units=neutral_H.units)
        neutral_H_stored.name = f"$M_{{\\rm HI + H_2}}$ ({aperture_size} kpc)"
        sphere.neutral_hydrogen_mass = neutral_H_stored

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

        H2_with_He = unyt.unyt_array(np.zeros(H2.shape), units=H2.units)
        H2_with_He[H > 0.0] = H2[H > 0.0] * (
            1.0 + He[H > 0.0] / H[H > 0.0]
        )
        H2_with_He.name = f"$M_{{\\rm H_2}}$ (incl. He, {aperture_size} kpc)"
        sphere.molecular_hydrogen_plus_helium_mass = H2_with_He

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

        jingle_select = M_star > unyt.unyt_quantity(10**8, "Solar_Mass")
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

    try:
        M_500_hot_gas = so500.hot_gas_mass
        f_hot_gas = M_500_hot_gas / M_500 / cosmic_baryon_fraction
        f_hot_gas.name = "$f_{\\rm hot\\,gas, 500, true} / (\\Omega_{\\rm b} / \\Omega_{\\rm m})$"
        so500.hot_gas_fraction = f_hot_gas
    except AttributeError:
        pass


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


def register_stellar_magnitudes(aperture_sizes, band_columns):
    for aperture_size in aperture_sizes:
        sphere = getattr(soap, f"exclusive_sphere_{aperture_size}kpc")
        try:
            lum = sphere.stellar_luminosity
            for band, col in band_columns.items():
                L_AB = lum[:, col]
                m_AB = np.copy(L_AB.value)
                mask = m_AB > 0.0
                m_AB[mask] = -2.5 * np.log10(m_AB[mask])
                mag = unyt.unyt_array(m_AB, units="dimensionless")
                mag.name = f"{band}-band AB magnitudes ({aperture_size} kpc)"
                setattr(sphere, f"stellar_magnitude_{band}_band", mag)
        except AttributeError:
            pass


def register_snia_rates_per_stellar_mass(aperture_sizes):
    for aperture_size in aperture_sizes:
        sphere = getattr(soap, f"exclusive_sphere_{aperture_size}kpc")
        try:
            snia_rate = sphere.total_snia_rate
            stellar_mass = sphere.stellar_mass
            rate_per_mass = unyt.unyt_array(
                np.zeros(len(stellar_mass)), units=snia_rate.units / stellar_mass.units
            )
            mask = stellar_mass > unyt.unyt_quantity(0.0, stellar_mass.units)
            rate_per_mass[mask] = (snia_rate[mask] / stellar_mass[mask]).to(
                snia_rate.units / stellar_mass.units
            )
            rate_per_mass.name = f"SNIa Rate / $M_*$ ({aperture_size} kpc)"
            sphere.snia_rate_per_stellar_mass = rate_per_mass
        except AttributeError:
            pass


def register_stellar_mass_selection_masks(aperture_sizes):
    thresholds = [
        (1e9, "1e9"),
        (1e10, "1e10"),
        (5e10, "5e10"),
    ]
    for aperture_size in aperture_sizes:
        sphere = getattr(soap, f"exclusive_sphere_{aperture_size}kpc")
        stellar_mass = sphere.stellar_mass
        is_active = np.array(sphere.is_active, dtype=bool)
        for threshold, label in thresholds:
            above = np.array(
                stellar_mass > unyt.unyt_quantity(threshold, "Solar_Mass"), dtype=bool
            )
            above_arr = unyt.unyt_array(above.astype(float), "dimensionless")
            above_arr.name = f"$M_* > {label}$ $M_\\odot$ ({aperture_size} kpc)"
            setattr(sphere, f"stellar_mass_above_{label}_msun", above_arr)

            above_active = np.logical_and(above, is_active)
            above_active_arr = unyt.unyt_array(above_active.astype(float), "dimensionless")
            above_active_arr.name = f"$M_* > {label}$ $M_\\odot$ and active ({aperture_size} kpc)"
            setattr(sphere, f"stellar_mass_above_{label}_msun_and_active", above_active_arr)


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
register_magnesium_to_hydrogen(aperture_sizes, solar_mg_abundance)
register_star_mg_and_o_to_fe(aperture_sizes)
register_dust(aperture_sizes)
register_neutral_gas_fractions(aperture_sizes)
register_species_fractions(aperture_sizes)
register_stellar_birth_density()
register_gas_fraction()
register_los_stellar_velocity_dispersion()
register_stellar_magnitudes(aperture_sizes, BAND_COLUMNS)
register_snia_rates_per_stellar_mass(aperture_sizes)
register_stellar_mass_selection_masks(aperture_sizes)
register_stellar_mass_scatter(stellar_mass_scatter_amplitude)
