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
    + LOS stellar velocity dispersions (10, 30 kpc) (los_veldisp_star_{x}_kpc)
        The LOS velocity dispersion, obtained by multiplying the 3D velocity
        dispersion with 1/sqrt(3).
    + mass_star_with_scatter_50_kpc
        Stellar mass with an additional 0.3 dex log-normal scatter.
    + stellar_mass_is_bigger_than_1e10_msun (30, 50, 100 kpc)
        Is the stellar mass larger than 1e10 Msun in the given aperture
"""

# Define aperture size in kpc
aperture_sizes_30_50_100_kpc = {30, 50, 100}
aperture_sizes_10_30_50_100_kpc = {10, 30, 50, 100}

# Solar metal mass fraction used in Ploeckinger S. & Schaye J. (2020)
solar_metal_mass_fraction = 0.0134

# Solar value for O/H
twelve_plus_log_OH_solar = 8.69

# Solar Fe abundance (from Wiersma et al 2009a)
solar_fe_abundance = 2.82e-5

# Additional scatter in stellar mass (in dex)
stellar_mass_scatter_amplitude = 0.3


def register_specific_star_formation_rates(self, catalogue, aperture_sizes):

    # Lowest sSFR below which the galaxy is considered passive
    marginal_ssfr = unyt.unyt_quantity(1e-11, units=1 / unyt.year)

    # Loop over apertures
    for aperture_size in aperture_sizes:

        # Get the halo mass
        halo_mass = catalogue.get_quantity("masses.mass_200crit")

        # Get stellar mass
        stellar_mass = catalogue.get_quantity(
            f"apertures.mass_star_{aperture_size}_kpc"
        )
        # Need to mask out zeros, otherwise we get RuntimeWarnings
        good_stellar_mass = stellar_mass > unyt.unyt_quantity(0.0, stellar_mass.units)

        # Get star formation rate
        star_formation_rate = catalogue.get_quantity(
            f"apertures.sfr_gas_{aperture_size}_kpc"
        )

        # Compute specific star formation rate using the "good" stellar mass
        ssfr = unyt.unyt_array(np.zeros(len(star_formation_rate)), units=1 / unyt.year)
        ssfr[good_stellar_mass] = (
            star_formation_rate[good_stellar_mass] / stellar_mass[good_stellar_mass]
        ).to(ssfr.units)

        # Name (label) of the derived field
        ssfr.name = f"Specific SFR ({aperture_size} kpc)"

        # Mask for the passive objects
        is_passive = unyt.unyt_array(
            (ssfr < (1.01 * marginal_ssfr).to(ssfr.units)).astype(float),
            units="dimensionless",
        )
        is_passive.name = "Passive Fraction"

        # Mask for the active objects
        is_active = unyt.unyt_array(
            (ssfr > (1.01 * marginal_ssfr).to(ssfr.units)).astype(float),
            units="dimensionless",
        )
        is_active.name = "Active Fraction"

        # Mask for active galaxies above 10^9 Msun
        is_bigger_than_1e9_active = unyt.unyt_array(
            (
                (stellar_mass > (1e9 * unyt.Msun).to(stellar_mass.units))
                & (ssfr > (1.01 * marginal_ssfr).to(ssfr.units))
            ).astype(float),
            units="dimensionless",
        )
        is_bigger_than_1e9_active.name = "Stellar mass larger than 10^9 Msun and active"

        # Mask for galaxies above 10^10 Msun
        is_bigger_than_1e10 = unyt.unyt_array(
            (stellar_mass > unyt.unyt_quantity(1e10, units="Msun")).astype(float),
            units="dimensionless",
        )
        is_bigger_than_1e10.name = "Stellar mass larger than 10^10 Msun"

        # Mask for active galaxies above 10^10 Msun
        is_bigger_than_1e10_active = unyt.unyt_array(
            (
                (stellar_mass > (1e10 * unyt.Msun).to(stellar_mass.units))
                & (ssfr > (1.01 * marginal_ssfr).to(ssfr.units))
            ).astype(float),
            units="dimensionless",
        )
        is_bigger_than_1e10_active.name = (
            "Stellar mass larger than 10^10 Msun and active"
        )

        # Mask for galaxies above 5* 10^10 Msun
        is_bigger_than_5e10 = unyt.unyt_array(
            (stellar_mass > (5e10 * unyt.Msun).to(stellar_mass.units)).astype(float),
            units="dimensionless",
        )
        is_bigger_than_5e10.name = "Stellar mass larger than 5 $\\times$ 10^10 Msun"

        # Mask for active galaxies above 5 * 10^10 Msun
        is_bigger_than_5e10_active = unyt.unyt_array(
            (
                (stellar_mass > (5e10 * unyt.Msun).to(stellar_mass.units))
                & (ssfr > (1.01 * marginal_ssfr).to(ssfr.units))
            ).astype(float),
            units="dimensionless",
        )
        is_bigger_than_5e10_active.name = (
            "Stellar mass larger than 5 $\\times$ 10^10 Msun and active"
        )

        # Get the specific star formation rate (per halo mass instead of stellar mass)
        sfr_M200 = unyt.unyt_array(
            np.zeros(star_formation_rate.shape),
            units=star_formation_rate.units / halo_mass.units,
        )
        sfr_M200[halo_mass > 0.0] = (
            star_formation_rate[halo_mass > 0.0] / halo_mass[halo_mass > 0.0]
        )
        sfr_M200.name = "Star formation rate divided by halo mass"

        # Register derived fields with specific star formation rates
        setattr(self, f"specific_sfr_gas_{aperture_size}_kpc", ssfr)
        setattr(self, f"is_passive_{aperture_size}_kpc", is_passive)
        setattr(self, f"is_active_{aperture_size}_kpc", is_active)
        setattr(self, f"sfr_halo_mass_{aperture_size}_kpc", sfr_M200)
        setattr(
            self,
            f"stellar_mass_is_bigger_than_1e10_msun_{aperture_size}_kpc",
            is_bigger_than_1e10,
        )
        setattr(
            self,
            f"stellar_mass_is_bigger_than_1e9_msun_active_{aperture_size}_kpc",
            is_bigger_than_1e9_active,
        )
        setattr(
            self,
            f"stellar_mass_is_bigger_than_1e10_msun_active_{aperture_size}_kpc",
            is_bigger_than_1e10_active,
        )
        setattr(
            self,
            f"stellar_mass_is_bigger_than_5e10_msun_{aperture_size}_kpc",
            is_bigger_than_5e10,
        )
        setattr(
            self,
            f"stellar_mass_is_bigger_than_5e10_msun_active_{aperture_size}_kpc",
            is_bigger_than_5e10_active,
        )

    return


def register_star_metallicities(self, catalogue, aperture_sizes, Z_sun):

    # Loop over apertures
    for aperture_size in aperture_sizes:

        try:
            metal_mass_fraction_star = (
                catalogue.get_quantity(f"apertures.zmet_star_{aperture_size}_kpc")
                / Z_sun
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

                L_AB = catalogue.get_quantity(
                    f"stellar_luminosities.{band}_luminosity_{aperture_size}_kpc"
                )
                m_AB = np.copy(L_AB)
                mask = L_AB > 0.0
                m_AB[mask] = -2.5 * np.log10(m_AB[mask])
                m_AB = unyt.unyt_array(m_AB, units="dimensionless")
                m_AB.name = f"{band}-band AB magnitudes ({aperture_size} kpc)"
                setattr(self, f"magnitudes_{band}_band_{aperture_size}_kpc", m_AB)

            except AttributeError:
                pass

    return

def register_corrected_star_magnitudes(self, catalogue, aperture_sizes, add_dust = False):

    bands = ["r", "u", "z", "K", "FUV"]

    if add_dust == True:
        dust = "w_dust"
    else:
        dust = ""
        
    # Loop over apertures
    for aperture_size in aperture_sizes:
        for band in bands:
            try:

                L_AB = catalogue.get_quantity(
                    f"corrected_stellar_luminosities{dust}.{band}_luminosity_{aperture_size}_kpc"
                )
                m_AB = np.copy(L_AB)
                mask = L_AB > 0.0
                m_AB[mask] = -2.5 * np.log10(m_AB[mask])
                m_AB = unyt.unyt_array(m_AB, units="dimensionless")
                m_AB.name = f"{band}-band AB magnitudes ({aperture_size} kpc)"
                setattr(self, f"corrected_magnitudes{dust}_{band}_band_{aperture_size}_kpc", m_AB)

            except AttributeError:
                pass

    return


def register_chabrier_masses(self, catalogue, aperture_sizes):

    # Loop over apertures
    for aperture_size in aperture_sizes:
        try:
            m_chab = catalogue.get_quantity(
                f"masses.chabrier_stellar_masses_{aperture_size}_kpc"
            )
            m_chab = unyt.unyt_array(m_chab, units="Msun")
            m_chab.name = f"Chabrier-IMF inferred stellar mass ({aperture_size} kpc)"
            setattr(self, f"chabrier_stellar_masses_{aperture_size}_kpc", m_chab)

        except AttributeError:
            pass

    return


def register_stellar_to_halo_mass_ratios(self, catalogue, aperture_sizes):

    # Loop over apertures
    for aperture_size in aperture_sizes:

        # Get the stellar mass in the aperture of a given size
        stellar_mass = catalogue.get_quantity(
            f"apertures.mass_star_{aperture_size}_kpc"
        )

        # M200 critical
        halo_M200crit = catalogue.get_quantity("masses.mass_200crit")
        smhm = unyt.unyt_array(
            np.zeros(stellar_mass.shape), units=stellar_mass.units / halo_M200crit.units
        )
        smhm[halo_M200crit > 0.0] = (
            stellar_mass[halo_M200crit > 0.0] / halo_M200crit[halo_M200crit > 0.0]
        )
        smhm.name = f"$M_* / M_{{\\rm 200crit}}$ ({aperture_size} kpc)"
        setattr(self, f"stellar_mass_to_halo_mass_200crit_{aperture_size}_kpc", smhm)

        # BN98
        halo_MBN98 = catalogue.get_quantity("masses.mass_bn98")
        smhm = unyt.unyt_array(
            np.zeros(stellar_mass.shape), units=stellar_mass.units / halo_MBN98.units
        )
        smhm[halo_MBN98 > 0.0] = (
            stellar_mass[halo_MBN98 > 0.0] / halo_MBN98[halo_MBN98 > 0.0]
        )
        smhm.name = f"$M_* / M_{{\\rm BN98}}$ ({aperture_size} kpc)"
        setattr(self, f"stellar_mass_to_halo_mass_bn98_{aperture_size}_kpc", smhm)

    return


def register_projected_stellar_masses(self, catalogue, aperture_sizes):

    # Loop over apertures
    for aperture_size in aperture_sizes:

        # Get the stellar mass in the aperture of a given size, and calculate
        # an average over the three axes
        stellar_mass = (
            catalogue.get_quantity(
                f"projected_apertures.projected_1_mass_star_{aperture_size}_kpc"
            )
            + catalogue.get_quantity(
                f"projected_apertures.projected_2_mass_star_{aperture_size}_kpc"
            )
            + catalogue.get_quantity(
                f"projected_apertures.projected_3_mass_star_{aperture_size}_kpc"
            )
        ) / 3

        stellar_mass.name = f"$M_*$ (2D, {aperture_size} kpc)"
        setattr(self, f"projected_stellar_mass_{aperture_size}_kpc", stellar_mass)

    return


def register_dust(self, catalogue, aperture_sizes, Z_sun, twelve_plus_log_OH_solar):
    # Loop over apertures
    for aperture_size in aperture_sizes:
        metal_frac = catalogue.get_quantity(f"apertures.zmet_gas_{aperture_size}_kpc")
        try:
            # Fetch dust fields
            dust_mass_silicates = catalogue.get_quantity(
                f"dust_masses.silicates_mass_{aperture_size}_kpc"
            )
            dust_mass_graphite = catalogue.get_quantity(
                f"dust_masses.graphite_mass_{aperture_size}_kpc"
            )
            dust_mass_large_grain = catalogue.get_quantity(
                f"dust_masses.large_grain_mass_{aperture_size}_kpc"
            )
            dust_mass_small_grain = catalogue.get_quantity(
                f"dust_masses.small_grain_mass_{aperture_size}_kpc"
            )
            dust_mass_silicates_hi = catalogue.get_quantity(
                f"dust_masses.atomic_silicates_mass_{aperture_size}_kpc"
            )
            dust_mass_graphite_hi = catalogue.get_quantity(
                f"dust_masses.atomic_graphite_mass_{aperture_size}_kpc"
            )
            dust_mass_silicates_h2 = catalogue.get_quantity(
                f"dust_masses.molecular_silicates_mass_{aperture_size}_kpc"
            )
            dust_mass_graphite_h2 = catalogue.get_quantity(
                f"dust_masses.molecular_graphite_mass_{aperture_size}_kpc"
            )
            dust_mass_large_grain_h2 = catalogue.get_quantity(
                f"dust_masses.molecular_large_grain_mass_{aperture_size}_kpc"
            )
            dust_mass_small_grain_h2 = catalogue.get_quantity(
                f"dust_masses.molecular_small_grain_mass_{aperture_size}_kpc"
            )
            dust_mass_silicates_cd = catalogue.get_quantity(
                f"dust_masses.cold_dense_silicates_mass_{aperture_size}_kpc"
            )
            dust_mass_graphite_cd = catalogue.get_quantity(
                f"dust_masses.cold_dense_graphite_mass_{aperture_size}_kpc"
            )
            dust_mass_large_grain_cd = catalogue.get_quantity(
                f"dust_masses.cold_dense_large_grain_mass_{aperture_size}_kpc"
            )
            dust_mass_small_grain_cd = catalogue.get_quantity(
                f"dust_masses.cold_dense_small_grain_mass_{aperture_size}_kpc"
            )

            # All dust mass
            dust_mass_total = dust_mass_graphite + dust_mass_silicates
            dust_mass_total_hi = dust_mass_graphite_hi + dust_mass_silicates_hi
            dust_mass_total_h2 = dust_mass_graphite_h2 + dust_mass_silicates_h2
            dust_mass_total_cd = dust_mass_graphite_cd + dust_mass_silicates_cd
            dust_mass_total_neutral = dust_mass_graphite + dust_mass_silicates

            small_to_large = unyt.unyt_array(
                np.zeros(dust_mass_small_grain.shape), units=unyt.dimensionless
            )
            small_to_large[dust_mass_large_grain > 0.0] = (
                dust_mass_small_grain[dust_mass_large_grain > 0.0]
                / dust_mass_large_grain[dust_mass_large_grain > 0.0]
            )
            small_to_large_h2 = unyt.unyt_array(
                np.zeros(dust_mass_small_grain_h2.shape), units=unyt.dimensionless
            )
            small_to_large_h2[dust_mass_large_grain_h2 > 0.0] = (
                dust_mass_small_grain_h2[dust_mass_large_grain_h2 > 0.0]
                / dust_mass_large_grain_h2[dust_mass_large_grain_h2 > 0.0]
            )
            small_to_large_cd = unyt.unyt_array(
                np.zeros(dust_mass_small_grain_cd.shape), units=unyt.dimensionless
            )
            small_to_large_cd[dust_mass_large_grain_cd > 0.0] = (
                dust_mass_small_grain_cd[dust_mass_large_grain_cd > 0.0]
                / dust_mass_large_grain_cd[dust_mass_large_grain_cd > 0.0]
            )

        # In case run without dust
        except AttributeError:
            dust_mass_large_grain = unyt.unyt_array(
                np.zeros_like(metal_frac), units="Msun"
            )
            dust_mass_large_grain_h2 = unyt.unyt_array(
                np.zeros_like(metal_frac), units="Msun"
            )
            dust_mass_large_grain_cd = unyt.unyt_array(
                np.zeros_like(metal_frac), units="Msun"
            )
            dust_mass_small_grain = unyt.unyt_array(
                np.zeros_like(metal_frac), units="Msun"
            )
            dust_mass_small_grain_h2 = unyt.unyt_array(
                np.zeros_like(metal_frac), units="Msun"
            )
            dust_mass_small_grain_cd = unyt.unyt_array(
                np.zeros_like(metal_frac), units="Msun"
            )
            dust_mass_total = unyt.unyt_array(np.zeros_like(metal_frac), units="Msun")
            dust_mass_total_hi = unyt.unyt_array(
                np.zeros_like(metal_frac), units="Msun"
            )
            dust_mass_total_h2 = unyt.unyt_array(
                np.zeros_like(metal_frac), units="Msun"
            )
            dust_mass_total_cd = unyt.unyt_array(
                np.zeros_like(metal_frac), units="Msun"
            )
            dust_mass_total_neutral = unyt.unyt_array(
                np.zeros_like(metal_frac), units="Msun"
            )
            small_to_large = unyt.unyt_array(
                np.zeros_like(metal_frac), units="dimensionless"
            )
            small_to_large_h2 = unyt.unyt_array(
                np.zeros_like(metal_frac), units="dimensionless"
            )
            small_to_large_cd = unyt.unyt_array(
                np.zeros_like(metal_frac), units="dimensionless"
            )

        dust_mass_total_hi.name = f"$M_{{\\rm dust,HI}}$ ({aperture_size} kpc)"
        dust_mass_total_h2.name = f"$M_{{\\rm dust,H2}}$ ({aperture_size} kpc)"
        dust_mass_total_cd.name = f"$M_{{\\rm dust,CD}}$ ({aperture_size} kpc)"
        dust_mass_total_neutral.name = f"$M_{{\\rm dust,Neut.}}$ ({aperture_size} kpc)"

        small_to_large_h2.name = f"$\\mathcal{{D}}_{{\\rm S}} / \\mathcal{{D}}_{{\\rm L}}$ ({aperture_size} kpc)"
        small_to_large_cd.name = f"$\\mathcal{{D}}_{{\\rm S,CD}} / \\mathcal{{D}}_{{\\rm L,CD}}$ ({aperture_size} kpc)"
        small_to_large.name = f"$\\mathcal{{D}}_{{\\rm S,Neut.}} / \\mathcal{{D}}_{{\\rm L,Neut.}}$ ({aperture_size} kpc)"

        # Fetch gas masses
        gas_mass = catalogue.get_quantity(f"apertures.mass_gas_{aperture_size}_kpc")
        atomic_mass = catalogue.get_quantity(
            f"gas_hydrogen_species_masses.HI_mass_{aperture_size}_kpc"
        )
        molecular_mass = catalogue.get_quantity(
            f"gas_hydrogen_species_masses.H2_mass_{aperture_size}_kpc"
        )
        neutral_mass = atomic_mass + molecular_mass
        colddense_mass = catalogue.get_quantity(
            f"cold_dense_gas_properties.cold_dense_gas_mass_{aperture_size}_kpc"
        )

        # Metal mass fractions of the gas

        try:
            linOH_abundance_times_mgas = catalogue.get_quantity(
                f"lin_element_ratios_times_masses.lin_O_over_H_total_times_gas_mass_100_kpc"
            )

            logOH_abundance_times_mhi = catalogue.get_quantity(
                f"log_element_ratios_times_masses.log_O_over_H_atomic_times_gas_mass_lowfloor_{aperture_size}_kpc"
            )
            logOH_abundance_times_mh2 = catalogue.get_quantity(
                f"log_element_ratios_times_masses.log_O_over_H_molecular_times_gas_mass_lowfloor_{aperture_size}_kpc"
            )
            logOH_abundance_times_cd = catalogue.get_quantity(
                f"log_element_ratios_times_masses.log_O_over_H_times_gas_mass_lowfloor_{aperture_size}_kpc"
            )
        except AttributeError:
            linOH_abundance_times_mgas = unyt.unyt_array(
                np.zeros_like(metal_frac), units="Msun"
            )
            logOH_abundance_times_mhi = unyt.unyt_array(
                np.zeros_like(metal_frac), units="Msun"
            )
            logOH_abundance_times_mh2 = unyt.unyt_array(
                np.zeros_like(metal_frac), units="Msun"
            )
            logOH_abundance_times_cd = unyt.unyt_array(
                np.zeros_like(metal_frac), units="Msun"
            )

        metal_frac_gas = unyt.unyt_array(
            np.zeros(colddense_mass.shape), units=unyt.dimensionless
        )
        mask_positive = (linOH_abundance_times_mgas > 0.0) & (colddense_mass > 0.0)
        metal_frac_gas[mask_positive] = (
            pow(
                10,
                np.log10(
                    linOH_abundance_times_mgas[mask_positive]
                    / colddense_mass[mask_positive]
                )
                + 12
                - twelve_plus_log_OH_solar,
            )
            * Z_sun
        )
        metal_frac_hi_fromO = unyt.unyt_array(
            np.zeros(atomic_mass.shape), units=unyt.dimensionless
        )
        metal_frac_hi_fromO[atomic_mass > 0.0] = (
            pow(
                10,
                (
                    (
                        logOH_abundance_times_mhi[atomic_mass > 0.0]
                        / atomic_mass[atomic_mass > 0.0]
                    )
                    + 12
                )
                - twelve_plus_log_OH_solar,
            )
            * Z_sun
        )
        metal_frac_h2_fromO = unyt.unyt_array(
            np.zeros(molecular_mass.shape), units=unyt.dimensionless
        )
        metal_frac_h2_fromO[molecular_mass > 0.0] = (
            pow(
                10,
                (
                    (
                        logOH_abundance_times_mh2[molecular_mass > 0.0]
                        / molecular_mass[molecular_mass > 0.0]
                    )
                    + 12
                )
                - twelve_plus_log_OH_solar,
            )
            * Z_sun
        )
        metal_frac_neutral_fromO = unyt.unyt_array(
            np.zeros(neutral_mass.shape), units=unyt.dimensionless
        )
        metal_frac_neutral_fromO[neutral_mass > 0.0] = (
            (metal_frac_hi_fromO[neutral_mass > 0.0] * atomic_mass[neutral_mass > 0.0])
            + (
                metal_frac_h2_fromO[neutral_mass > 0.0]
                * molecular_mass[neutral_mass > 0.0]
            )
        ) / neutral_mass[neutral_mass > 0.0]
        metal_frac_cd_fromO = unyt.unyt_array(
            np.zeros(atomic_mass.shape), units=unyt.dimensionless
        )
        metal_frac_cd_fromO[atomic_mass > 0.0] = (
            pow(
                10,
                (
                    (
                        logOH_abundance_times_cd[atomic_mass > 0.0]
                        / atomic_mass[atomic_mass > 0.0]
                    )
                    + 12
                )
                - twelve_plus_log_OH_solar,
            )
            * Z_sun
        )
        # Add label to the dust mass field
        dust_mass_total.name = f"$M_{{\\rm dust}}$ ({aperture_size} kpc)"

        # Compute dust to gas fraction
        dust_to_gas = unyt.unyt_array(
            np.zeros(gas_mass.shape), units=unyt.dimensionless
        )
        dust_to_gas[gas_mass > 0.0] = (
            dust_mass_total[gas_mass > 0.0] / gas_mass[gas_mass > 0.0]
        )
        dust_to_gas_hi = unyt.unyt_array(
            np.zeros(atomic_mass.shape), units=unyt.dimensionless
        )
        dust_to_gas_hi[atomic_mass > 0.0] = (
            dust_mass_total_hi[atomic_mass > 0.0] / atomic_mass[atomic_mass > 0.0]
        )
        dust_to_gas_h2 = unyt.unyt_array(
            np.zeros(molecular_mass.shape), units=unyt.dimensionless
        )
        dust_to_gas_h2[molecular_mass > 0.0] = (
            dust_mass_total_h2[molecular_mass > 0.0]
            / molecular_mass[molecular_mass > 0.0]
        )
        dust_to_gas_neutral = unyt.unyt_array(
            np.zeros(neutral_mass.shape), units=unyt.dimensionless
        )
        dust_to_gas_neutral[neutral_mass > 0.0] = (
            dust_mass_total_hi[neutral_mass > 0.0]
            + dust_mass_total_h2[neutral_mass > 0.0]
        ) / neutral_mass[neutral_mass > 0.0]
        dust_to_gas_cd = unyt.unyt_array(
            np.zeros(colddense_mass.shape), units=unyt.dimensionless
        )
        dust_to_gas_cd[colddense_mass > 0.0] = (
            dust_mass_total_cd[colddense_mass > 0.0]
            / colddense_mass[colddense_mass > 0.0]
        )

        # Label for the dust-fraction derived field
        dust_to_gas.name = f"$\\mathcal{{DTG}}$ ({aperture_size} kpc)"
        dust_to_gas_hi.name = f"$\\mathcal{{DTG}}$ (atomic phase, {aperture_size} kpc)"
        dust_to_gas_h2.name = (
            f"$\\mathcal{{DTG}}$ (molecular phase, {aperture_size} kpc)"
        )
        dust_to_gas_neutral.name = (
            f"$\\mathcal{{DTG}}$ (neutral phase, {aperture_size} kpc)"
        )
        dust_to_gas_cd.name = (
            f"$\\mathcal{{DTG}}$ (cold, dense phase, {aperture_size} kpc)"
        )

        # Compute to metal ratio
        dust_to_metals = unyt.unyt_array(
            np.zeros(metal_frac.shape), units=unyt.dimensionless
        )
        dust_to_metals[metal_frac > 0.0] = (
            dust_to_gas[metal_frac > 0.0] / metal_frac[metal_frac > 0.0]
        )
        dust_to_metals_hi = unyt.unyt_array(
            np.zeros(metal_frac_hi_fromO.shape), units=unyt.dimensionless
        )
        dust_to_metals_hi[metal_frac_hi_fromO > 0.0] = (
            dust_to_gas_hi[metal_frac_hi_fromO > 0.0]
            / metal_frac_hi_fromO[metal_frac_hi_fromO > 0.0]
        )
        dust_to_metals_h2 = unyt.unyt_array(
            np.zeros(metal_frac_h2_fromO.shape), units=unyt.dimensionless
        )
        dust_to_metals_h2[metal_frac_h2_fromO > 0.0] = (
            dust_to_gas_h2[metal_frac_h2_fromO > 0.0]
            / metal_frac_h2_fromO[metal_frac_h2_fromO > 0.0]
        )
        dust_to_metals_neutral = unyt.unyt_array(
            np.zeros(metal_frac_neutral_fromO.shape), units=unyt.dimensionless
        )
        dust_to_metals_neutral[metal_frac_neutral_fromO > 0.0] = (
            dust_to_gas_neutral[metal_frac_neutral_fromO > 0.0]
            / metal_frac_neutral_fromO[metal_frac_neutral_fromO > 0.0]
        )
        dust_to_metals_cd = unyt.unyt_array(
            np.zeros(metal_frac_gas.shape), units=unyt.dimensionless
        )
        dust_to_metals_cd[metal_frac_gas > 0.0] = (
            dust_to_gas_cd[metal_frac_gas > 0.0] / metal_frac_gas[metal_frac_gas > 0.0]
        )  # metal_frac_cd_fromO

        dust_to_metals.name = f"$\\mathcal{{DTM}}$ ({aperture_size} kpc)"
        dust_to_metals_hi.name = (
            f"$\\mathcal{{DTM}}$ (atomic phase, {aperture_size} kpc)"
        )
        dust_to_metals_h2.name = (
            f"$\\mathcal{{DTM}}$ (molecular phase, {aperture_size} kpc)"
        )
        dust_to_metals_neutral.name = (
            f"$\\mathcal{{DTM}}$ (neutral phase, {aperture_size} kpc)"
        )
        dust_to_metals_cd.name = (
            f"$\\mathcal{{DTM}}$ (cold, dense phase, {aperture_size} kpc)"
        )

        # Compute dust to stellar ratio
        mass_star = catalogue.get_quantity(f"apertures.mass_star_{aperture_size}_kpc")
        dust_to_stars = unyt.unyt_array(
            np.zeros(mass_star.shape), units=unyt.dimensionless
        )
        dust_to_stars[mass_star > 0.0] = (
            dust_mass_total[mass_star > 0.0] / mass_star[mass_star > 0.0]
        )
        dust_to_stars_hi = unyt.unyt_array(
            np.zeros(mass_star.shape), units=unyt.dimensionless
        )
        dust_to_stars_hi[mass_star > 0.0] = (
            dust_mass_total_hi[mass_star > 0.0] / mass_star[mass_star > 0.0]
        )
        dust_to_stars_h2 = unyt.unyt_array(
            np.zeros(mass_star.shape), units=unyt.dimensionless
        )
        dust_to_stars_h2[mass_star > 0.0] = (
            dust_mass_total_h2[mass_star > 0.0] / mass_star[mass_star > 0.0]
        )
        dust_to_stars_neutral = unyt.unyt_array(
            np.zeros(mass_star.shape), units=unyt.dimensionless
        )
        dust_to_stars_neutral[mass_star > 0.0] = (
            dust_mass_total_neutral[mass_star > 0.0] / mass_star[mass_star > 0.0]
        )
        dust_to_stars_cd = unyt.unyt_array(
            np.zeros(mass_star.shape), units=unyt.dimensionless
        )
        dust_to_stars_cd[mass_star > 0.0] = (
            dust_mass_total_cd[mass_star > 0.0] / mass_star[mass_star > 0.0]
        )

        dust_to_stars.name = f"$M_{{\\rm dust}}/M_*$ ({aperture_size} kpc)"
        dust_to_stars_hi.name = (
            f"$M_{{\\rm dust}}/M_*$ (atomic phase, {aperture_size} kpc)"
        )
        dust_to_stars_h2.name = (
            f"$M_{{\\rm dust}}/M_*$ (molecular phase, {aperture_size} kpc)"
        )
        dust_to_stars_neutral.name = (
            f"$M_{{\\rm dust}}/M_*$ (neutral phase, {aperture_size} kpc)"
        )
        dust_to_stars_cd.name = (
            f"$M_{{\\rm dust}}/M_*$ (cold, dense phase, {aperture_size} kpc)"
        )

        setattr(
            self,
            f"jingle_galaxy_selection_{aperture_size}kpc",
            mass_star > unyt.unyt_quantity(10 ** 8, "Solar_Mass"),
        )

        setattr(
            self,
            f"has_sizes_{aperture_size}kpc",
            np.logical_and(dust_mass_large_grain > 0, dust_mass_small_grain > 0),
        )
        setattr(
            self,
            f"has_sizes_h2_{aperture_size}kpc",
            np.logical_and(dust_mass_large_grain_h2 > 0, dust_mass_small_grain_h2 > 0),
        )
        setattr(
            self,
            f"has_sizes_cd_{aperture_size}kpc",
            np.logical_and(dust_mass_large_grain_cd > 0, dust_mass_small_grain_cd > 0),
        )

        setattr(
            self,
            f"jingle_galaxy_selection_{aperture_size}kpc",
            mass_star > unyt.unyt_quantity(10 ** 8, "Solar_Mass"),
        )

        # Register derived fields with dust
        setattr(self, f"total_dust_masses_{aperture_size}_kpc", dust_mass_total)
        setattr(self, f"dust_to_metal_ratio_{aperture_size}_kpc", dust_to_metals)
        setattr(self, f"dust_to_gas_ratio_{aperture_size}_kpc", dust_to_gas)
        setattr(self, f"dust_to_stellar_ratio_{aperture_size}_kpc", dust_to_stars)
        setattr(self, f"dust_small_to_large_ratio_{aperture_size}_kpc", small_to_large)

        setattr(
            self, f"total_atomic_dust_masses_{aperture_size}_kpc", dust_mass_total_hi
        )
        setattr(
            self, f"atomic_dust_to_metal_ratio_{aperture_size}_kpc", dust_to_metals_hi
        )
        setattr(self, f"atomic_dust_to_gas_ratio_{aperture_size}_kpc", dust_to_gas_hi)
        setattr(
            self, f"atomic_dust_to_stellar_ratio_{aperture_size}_kpc", dust_to_stars_hi
        )

        setattr(
            self, f"total_molecular_dust_masses_{aperture_size}_kpc", dust_mass_total_h2
        )
        setattr(
            self,
            f"molecular_dust_to_metal_ratio_{aperture_size}_kpc",
            dust_to_metals_h2,
        )
        setattr(
            self, f"molecular_dust_to_gas_ratio_{aperture_size}_kpc", dust_to_gas_h2
        )
        setattr(
            self,
            f"molecular_dust_to_stellar_ratio_{aperture_size}_kpc",
            dust_to_stars_h2,
        )
        setattr(
            self,
            f"molecular_dust_small_to_large_ratio_{aperture_size}_kpc",
            small_to_large_h2,
        )

        setattr(
            self,
            f"total_neutral_dust_masses_{aperture_size}_kpc",
            dust_mass_total_neutral,
        )
        setattr(
            self,
            f"neutral_dust_to_metal_ratio_{aperture_size}_kpc",
            dust_to_metals_neutral,
        )
        setattr(
            self, f"neutral_dust_to_gas_ratio_{aperture_size}_kpc", dust_to_gas_neutral
        )
        setattr(
            self,
            f"neutral_dust_to_stellar_ratio_{aperture_size}_kpc",
            dust_to_stars_neutral,
        )

        setattr(
            self,
            f"total_cold_dense_dust_masses_{aperture_size}_kpc",
            dust_mass_total_cd,
        )
        setattr(
            self,
            f"cold_dense_dust_to_metal_ratio_{aperture_size}_kpc",
            dust_to_metals_cd,
        )
        setattr(
            self, f"cold_dense_dust_to_gas_ratio_{aperture_size}_kpc", dust_to_gas_cd
        )
        setattr(
            self,
            f"cold_dense_dust_to_stellar_ratio_{aperture_size}_kpc",
            dust_to_stars_cd,
        )
        setattr(
            self,
            f"cold_dense_dust_small_to_large_ratio_{aperture_size}_kpc",
            small_to_large_cd,
        )

    return


def register_star_Mg_and_O_to_Fe(self, catalogue, aperture_sizes):

    # Ratio of solar abundancies (Asplund et al. 2009)
    X_O_to_X_Fe_solar = 4.44
    X_Mg_to_X_Fe_solar = 0.55

    # Loop over apertures
    for aperture_size in aperture_sizes:

        # Oxygen mass
        M_O = catalogue.get_quantity(
            f"element_masses_in_stars.oxygen_mass_{aperture_size}_kpc"
        )

        # Magnesium mass
        M_Mg = catalogue.get_quantity(
            f"element_masses_in_stars.magnesium_mass_{aperture_size}_kpc"
        )

        # Iron mass
        M_Fe = catalogue.get_quantity(
            f"element_masses_in_stars.iron_mass_{aperture_size}_kpc"
        )

        # Avoid zeroes
        mask_Mg = np.logical_and(M_Fe > 0.0 * M_Fe.units, M_Mg > 0.0 * M_Mg.units)
        mask_O = np.logical_and(M_Fe > 0.0 * M_Fe.units, M_O > 0.0 * M_O.units)

        # Floor value for the field below
        floor_value = -5

        Mg_over_Fe = floor_value * np.ones_like(M_Fe)
        Mg_over_Fe[mask_Mg] = np.log10(M_Mg[mask_Mg] / M_Fe[mask_Mg]) - np.log10(
            X_Mg_to_X_Fe_solar
        )
        O_over_Fe = floor_value * np.ones_like(M_Fe)
        O_over_Fe[mask_O] = np.log10(M_O[mask_O] / M_Fe[mask_O]) - np.log10(
            X_O_to_X_Fe_solar
        )

        # Convert to units used in observations
        Mg_over_Fe = unyt.unyt_array(Mg_over_Fe, "dimensionless")
        Mg_over_Fe.name = f"[Mg/Fe]$_*$ ({aperture_size} kpc)"
        O_over_Fe = unyt.unyt_array(O_over_Fe, "dimensionless")
        O_over_Fe.name = f"[O/Fe]$_*$ ({aperture_size} kpc)"

        # Register the field
        setattr(self, f"star_magnesium_over_iron_{aperture_size}_kpc", Mg_over_Fe)
        setattr(self, f"star_oxygen_over_iron_{aperture_size}_kpc", O_over_Fe)

    return


def register_nitrogen_to_oxygen(self, catalogue, aperture_sizes):
    # Loop over aperture average-of-linear N/O-abundances
    for aperture_size in aperture_sizes:

        for short_phase, long_phase in zip(
            ["_total", ""], ["Total (Diffuse + Dust)", "Diffuse"]
        ):

            # Fetch N over O times gas mass computed in apertures. The
            # mass ratio between N and O has already been accounted for.
            log_N_over_O_times_gas_mass = catalogue.get_quantity(
                f"lin_element_ratios_times_masses.lin_N_over_O{short_phase}_times_gas_mass_{aperture_size}_kpc"
            )
            # Fetch gas mass in apertures, here we are calling cold gas
            # that is part of the ISM and that is considered in the calculation of
            # lin_N_over_O_times_gas_mass
            gas_cold_dense_mass = catalogue.get_quantity(
                f"cold_dense_gas_properties.cold_dense_gas_mass_{aperture_size}_kpc"
            )

            # Compute gas-mass weighted O over H
            log_N_over_O = unyt.unyt_array(
                np.zeros_like(gas_cold_dense_mass), "dimensionless"
            )

            # Avoid division by zero
            mask = gas_cold_dense_mass > 0.0 * gas_cold_dense_mass.units
            log_N_over_O[mask] = np.log10(
                log_N_over_O_times_gas_mass[mask] / gas_cold_dense_mass[mask]
            )

            log_N_over_O.name = (
                f"{long_phase} Gas $\\log_{{10}}({{\\rm N/O}})$ ({aperture_size} kpc)"
            )

            # Register the field
            setattr(
                self,
                f"gas_n_over_o_abundance{short_phase}_avglin_{aperture_size}_kpc",
                log_N_over_O,
            )
            setattr(self, f"has_cold_dense_gas_{aperture_size}_kpc", mask)

        # register average-of-log O-abundances (high and low particle floors)
        for floor, floor_label in zip(
            ["low", "high"], ["Min = $10^{{-4}}$", "Min = $10^{{-3}}$"]
        ):
            # Fetch N over O times gas mass computed in apertures.
            # Note that here we are calling the diffuse quantities
            log_N_over_O_times_gas_mass = catalogue.get_quantity(
                f"log_element_ratios_times_masses.log_N_over_O_times_gas_mass_{floor}floor_{aperture_size}_kpc"
            )

            # Fetch gas mass in apertures
            gas_cold_dense_mass = catalogue.get_quantity(
                f"cold_dense_gas_properties.cold_dense_gas_mass_{aperture_size}_kpc"
            )

            # Compute gas-mass weighted N over O
            log_N_over_O = unyt.unyt_array(
                np.zeros_like(gas_cold_dense_mass), "dimensionless"
            )
            # Avoid division by zero
            mask = gas_cold_dense_mass > 0.0 * gas_cold_dense_mass.units
            log_N_over_O[mask] = (
                log_N_over_O_times_gas_mass[mask] / gas_cold_dense_mass[mask]
            )

            # Convert to units used in observations
            N_abundance = unyt.unyt_array(log_N_over_O, "dimensionless")
            N_abundance.name = f"Diffuse Gas $\\log_{{10}}({{\\rm N/O}})$ ({floor_label}, {aperture_size} kpc)"

            # Register the field
            setattr(
                self,
                f"gas_n_over_o_abundance_avglog_{floor}_{aperture_size}_kpc",
                N_abundance,
            )
            setattr(self, f"has_cold_dense_gas_{aperture_size}_kpc", mask)

    return


def register_carbon_to_oxygen(self, catalogue, aperture_sizes):
    # Loop over aperture average-of-linear C/O-abundances
    for aperture_size in aperture_sizes:

        for short_phase, long_phase in zip(
            ["_total", ""], ["Total (Diffuse + Dust)", "Diffuse"]
        ):
            # Fetch C over O times gas mass computed in apertures. The
            # mass ratio between N and O has already been accounted for.
            log_C_over_O_times_gas_mass = catalogue.get_quantity(
                f"lin_element_ratios_times_masses.lin_C_over_O{short_phase}_times_gas_mass_{aperture_size}_kpc"
            )

            # Fetch gas mass in apertures
            gas_cold_dense_mass = catalogue.get_quantity(
                f"cold_dense_gas_properties.cold_dense_gas_mass_{aperture_size}_kpc"
            )

            # Compute gas-mass weighted O over H
            log_C_over_O = unyt.unyt_array(
                np.zeros_like(gas_cold_dense_mass), "dimensionless"
            )
            # Avoid division by zero
            mask = gas_cold_dense_mass > 0.0 * gas_cold_dense_mass.units
            log_C_over_O[mask] = np.log10(
                log_C_over_O_times_gas_mass[mask] / gas_cold_dense_mass[mask]
            )

            log_C_over_O.name = (
                f"{long_phase} Gas $\\log_{{10}}({{\\rm C/O}})$ ({aperture_size} kpc)"
            )

            # Register the field
            setattr(
                self,
                f"gas_c_over_o_abundance{short_phase}_avglin_{aperture_size}_kpc",
                log_C_over_O,
            )
            setattr(self, f"has_cold_dense_gas_{aperture_size}_kpc", mask)

        # register average-of-log O-abundances (high and low particle floors)
        for floor, floor_label in zip(
            ["low", "high"], ["Min = $10^{{-4}}$", "Min = $10^{{-3}}$"]
        ):
            # Fetch C over O times gas mass computed in apertures.
            # Note that here we are calling the diffuse quantities.
            log_C_over_O_times_gas_mass = catalogue.get_quantity(
                f"log_element_ratios_times_masses.log_C_over_O_times_gas_mass_{floor}floor_{aperture_size}_kpc"
            )

            # Fetch gas mass in apertures
            gas_cold_dense_mass = catalogue.get_quantity(
                f"cold_dense_gas_properties.cold_dense_gas_mass_{aperture_size}_kpc"
            )

            # Compute gas-mass weighted O over H
            log_C_over_O = unyt.unyt_array(
                np.zeros_like(gas_cold_dense_mass), "dimensionless"
            )
            # Avoid division by zero
            mask = gas_cold_dense_mass > 0.0 * gas_cold_dense_mass.units
            log_C_over_O[mask] = (
                log_C_over_O_times_gas_mass[mask] / gas_cold_dense_mass[mask]
            )

            # Convert to units used in observations
            C_abundance = unyt.unyt_array(log_C_over_O, "dimensionless")
            C_abundance.name = f"Diffuse Gas $\\log_{{10}}({{\\rm C/O}})$ ({floor_label}, {aperture_size} kpc)"

            # Register the field
            setattr(
                self,
                f"gas_c_over_o_abundance_avglog_{floor}_{aperture_size}_kpc",
                C_abundance,
            )
            setattr(self, f"has_cold_dense_gas_{aperture_size}_kpc", mask)

    return


def register_oxygen_to_hydrogen(self, catalogue, aperture_sizes):
    # Loop over aperture average-of-linear O-abundances
    for aperture_size in aperture_sizes:
        # register linearly averaged O abundances
        for short_phase, long_phase in zip(
            ["_total", ""], ["Total (Diffuse + Dust)", "Diffuse"]
        ):
            # Fetch O over H times gas mass computed in apertures.  The factor of 16 (the
            # mass ratio between O and H) has already been accounted for.
            log_O_over_H_times_gas_mass = catalogue.get_quantity(
                f"lin_element_ratios_times_masses.lin_O_over_H{short_phase}_times_gas_mass_{aperture_size}_kpc"
            )
            # Fetch gas mass in apertures
            gas_cold_dense_mass = catalogue.get_quantity(
                f"cold_dense_gas_properties.cold_dense_gas_mass_{aperture_size}_kpc"
            )

            # Compute gas-mass weighted O over H
            log_O_over_H = unyt.unyt_array(
                np.zeros_like(gas_cold_dense_mass), "dimensionless"
            )
            # Avoid division by zero
            mask = gas_cold_dense_mass > 0.0 * gas_cold_dense_mass.units
            log_O_over_H[mask] = np.log10(
                log_O_over_H_times_gas_mass[mask] / gas_cold_dense_mass[mask]
            )

            # Convert to units used in observations
            O_abundance = unyt.unyt_array(12 + log_O_over_H, "dimensionless")
            O_abundance.name = f"SF {long_phase} Gas $12+\\log_{{10}}({{\\rm O/H}})$ ({aperture_size} kpc)"

            # Register the field
            setattr(
                self,
                f"gas_o_abundance{short_phase}_avglin_{aperture_size}_kpc",
                O_abundance,
            )
            setattr(self, f"has_cold_dense_gas_{aperture_size}_kpc", mask)

        # register average-of-log O-abundances (high and low particle floors)
        for floor, floor_label in zip(
            ["low", "high"], ["Min = $10^{{-4}}$", "Min = $10^{{-3}}$"]
        ):
            # Fetch O over H times gas mass computed in apertures.  The factor of 16 (the
            # mass ratio between O and H) has already been accounted for.
            log_O_over_H_times_gas_mass = catalogue.get_quantity(
                f"log_element_ratios_times_masses.log_O_over_H_times_gas_mass_{floor}floor_{aperture_size}_kpc"
            )

            # Fetch gas mass in apertures
            gas_cold_dense_mass = catalogue.get_quantity(
                f"cold_dense_gas_properties.cold_dense_gas_mass_{aperture_size}_kpc"
            )

            # Compute gas-mass weighted O over H
            log_O_over_H = unyt.unyt_array(
                np.zeros_like(gas_cold_dense_mass), "dimensionless"
            )
            # Avoid division by zero
            mask = gas_cold_dense_mass > 0.0 * gas_cold_dense_mass.units
            log_O_over_H[mask] = (
                log_O_over_H_times_gas_mass[mask] / gas_cold_dense_mass[mask]
            )

            # Convert to units used in observations
            O_abundance = unyt.unyt_array(12 + log_O_over_H, "dimensionless")
            O_abundance.name = f"SF Gas Diffuse $12+\\log_{{10}}({{\\rm O/H}})$ ({floor_label}, {aperture_size} kpc)"

            # Register the field
            setattr(
                self, f"gas_o_abundance_avglog_{floor}_{aperture_size}_kpc", O_abundance
            )

    return


def register_iron_to_hydrogen(self, catalogue, aperture_sizes, fe_solar_abundance):
    # Loop over apertures
    for aperture_size in aperture_sizes:
        # Fetch linear Fe over H times stellar mass computed in apertures. The
        # mass ratio between Fe and H has already been accounted for.
        lin_Fe_over_H_times_star_mass = catalogue.get_quantity(
            f"lin_element_ratios_times_masses.lin_Fe_over_H_times_star_mass_{aperture_size}_kpc"
        )
        # Fetch linear Fe (from SNIa) over H times stellar mass computed in apertures. The
        # mass ratio between Fe and H has already been accounted for.
        lin_FeSNIa_over_H_times_star_mass = catalogue.get_quantity(
            f"lin_element_ratios_times_masses.lin_FeSNIa_over_H_times_star_mass_{aperture_size}_kpc"
        )
        # Fetch stellar mass in apertures
        star_mass = catalogue.get_quantity(f"apertures.mass_star_{aperture_size}_kpc")

        # Compute stellar-mass weighted Fe over H
        Fe_over_H = unyt.unyt_array(np.zeros_like(star_mass), "dimensionless")
        # Avoid division by zero
        mask = star_mass > 0.0 * star_mass.units
        Fe_over_H[mask] = lin_Fe_over_H_times_star_mass[mask] / star_mass[mask]
        # Convert to units used in observations
        Fe_abundance = unyt.unyt_array(Fe_over_H / fe_solar_abundance, "dimensionless")
        Fe_abundance.name = f"Stellar (Fe/H) ({aperture_size} kpc)"

        # Register the field
        setattr(self, f"star_fe_abundance_avglin_{aperture_size}_kpc", Fe_abundance)

        # Compute stellar-mass weighted Fe over H
        FeSNIa_over_H = unyt.unyt_array(np.zeros_like(star_mass), "dimensionless")
        # Avoid division by zero
        mask = star_mass > 0.0 * star_mass.units
        FeSNIa_over_H[mask] = lin_FeSNIa_over_H_times_star_mass[mask] / star_mass[mask]
        # Convert to units used in observations
        FeSNIa_abundance = unyt.unyt_array(
            FeSNIa_over_H / fe_solar_abundance, "dimensionless"
        )
        FeSNIa_abundance.name = f"Stellar (Fe(SNIa)/H) ({aperture_size} kpc)"

        # Register the field
        setattr(
            self, f"star_fe_snia_abundance_avglin_{aperture_size}_kpc", FeSNIa_abundance
        )

        # register average-of-log Fe-abundances (high and low particle floors)
        for floor, floor_label in zip(
            ["low", "high"], ["Min = $10^{{-4}}$", "Min = $10^{{-3}}$"]
        ):

            # Fetch Fe over H times stellar mass computed in apertures. The
            # mass ratio between Fe and H has already been accounted for.
            log_Fe_over_H_times_star_mass = catalogue.get_quantity(
                f"log_element_ratios_times_masses.log_Fe_over_H_times_star_mass_{floor}floor_{aperture_size}_kpc"
            )
            # Fetch stellar mass in apertures
            star_mass = catalogue.get_quantity(
                f"apertures.mass_star_{aperture_size}_kpc"
            )

            # Compute stellar-mass weighted Fe over H
            Fe_over_H = unyt.unyt_array(np.zeros_like(star_mass), "dimensionless")
            # Avoid division by zero
            mask = star_mass > 0.0 * star_mass.units
            Fe_over_H[mask] = pow(
                10.0, log_Fe_over_H_times_star_mass[mask] / star_mass[mask]
            )
            # Convert to units used in observations
            Fe_abundance = unyt.unyt_array(
                Fe_over_H / fe_solar_abundance, "dimensionless"
            )
            Fe_abundance.name = (
                f"Stellar $10^{{\\rm [Fe/H]}}$ ({floor_label}, {aperture_size} kpc)"
            )

            log_Fe_over_H_times_star_mass = catalogue.get_quantity(
                f"log_element_ratios_times_masses.log_Fe_over_H_times_star_mass_{floor}floor_{aperture_size}_kpc"
            )

            # Register the field
            setattr(
                self,
                f"star_fe_abundance_avglog_{floor}_{aperture_size}_kpc",
                Fe_abundance,
            )

            if floor == "low":
                try:
                    log_Fe_over_H_times_star_mass = catalogue.get_quantity(
                        f"log_element_ratios_times_masses.log_SNIaFe_over_H_times_star_mass_{floor}floor_{aperture_size}_kpc"
                    )
                    Fe_over_H[mask] = pow(
                        10.0, log_Fe_over_H_times_star_mass[mask] / star_mass[mask]
                    )
                    # Convert to units used in observations
                    Fe_abundance = unyt.unyt_array(
                        Fe_over_H / fe_solar_abundance, "dimensionless"
                    )

                    Fe_abundance.name = f"Stellar $10^{{\\rm [Fe/H]}}$ ({floor_label}, {aperture_size} kpc)"
                    setattr(
                        self,
                        f"star_fe_snia_abundance_avglog_{floor}_{aperture_size}_kpc",
                        Fe_abundance,
                    )
                except AttributeError:
                    # else clip values to floor
                    setattr(
                        self,
                        f"star_fe_snia_abundance_avglog_{floor}_{aperture_size}_kpc",
                        unyt.unyt_array(
                            np.zeros_like(star_mass) - 4.0, "dimensionless"
                        ),
                    )
    return


def register_cold_dense_gas_metallicity(
    self, catalogue, aperture_sizes, Z_sun, log_twelve_plus_logOH_solar
):
    # Loop over apertures
    for aperture_size in aperture_sizes:
        # Fetch gas metal masses in apertures
        lin_diffuse_metallicity = catalogue.get_quantity(
            f"cold_dense_gas_properties.cold_dense_diffuse_metal_mass_{aperture_size}_kpc"
        )
        # Fetch gas mass in apertures
        gas_cold_dense_mass = catalogue.get_quantity(
            f"cold_dense_gas_properties.cold_dense_gas_mass_{aperture_size}_kpc"
        )

        # Compute gas-mass weighted metallicity, floor at a non-zero metallicity 1e-8
        twelve_plus_logOH = unyt.unyt_array(
            np.zeros_like(gas_cold_dense_mass) + 1e-8, "dimensionless"
        )
        # Avoid division by zero
        mask = gas_cold_dense_mass > 0.0 * gas_cold_dense_mass.units

        # convert absolute metallicity to 12+log10(O/H), assuming solar abundance patterns
        twelve_plus_logOH[mask] = (
            np.log10(
                lin_diffuse_metallicity[mask] / (Z_sun * gas_cold_dense_mass[mask])
            )
            + log_twelve_plus_logOH_solar
        )

        # Register the field
        O_abundance = unyt.unyt_array(twelve_plus_logOH, "dimensionless")
        O_abundance.name = (
            f"SF Gas $12+\\log_{{10}}({{\\rm O/H}})$ from $Z$ ({aperture_size} kpc)"
        )

        setattr(self, f"gas_o_abundance_fromz_avglin_{aperture_size}_kpc", O_abundance)

    return


def register_hi_masses(self, catalogue, aperture_sizes):

    # Loop over aperture sizes
    for aperture_size in aperture_sizes:

        # Fetch HI mass
        HI_mass = catalogue.get_quantity(
            f"gas_hydrogen_species_masses.HI_mass_{aperture_size}_kpc"
        )

        # Label of the derived field
        HI_mass.name = f"$M_{{\\rm HI}}$ ({aperture_size} kpc)"

        # Register derived field
        setattr(self, f"gas_HI_mass_{aperture_size}_kpc", HI_mass)

    return


def register_dust_to_hi_ratio(self, catalogue, aperture_sizes):

    # Loop over aperture sizes
    for aperture_size in aperture_sizes:

        # Fetch HI mass in apertures
        HI_mass = catalogue.get_quantity(
            f"gas_hydrogen_species_masses.HI_mass_{aperture_size}_kpc"
        )

        try:
            # Fetch dust fields
            dust_mass_silicates = catalogue.get_quantity(
                f"dust_masses.silicates_mass_{aperture_size}_kpc"
            )
            dust_mass_graphite = catalogue.get_quantity(
                f"dust_masses.graphite_mass_{aperture_size}_kpc"
            )
            # All dust mass
            dust_mass_total = dust_mass_graphite + dust_mass_silicates

        # In case run without dust
        except AttributeError:
            dust_mass_total = unyt.unyt_array(np.zeros_like(HI_mass), units="Msun")

        # Compute dust-to-HI mass ratios
        dust_to_hi_ratio = unyt.unyt_array(np.zeros_like(HI_mass), "dimensionless")
        # Avoid division by zero
        mask = HI_mass > 0.0 * HI_mass.units
        dust_to_hi_ratio[mask] = dust_mass_total[mask] / HI_mass[mask]

        # Add label
        dust_to_hi_ratio.name = f"M_{{\\rm dust}}/M_{{\\rm HI}} ({aperture_size} kpc)"

        # Register field
        setattr(self, f"gas_dust_to_hi_ratio_{aperture_size}_kpc", dust_to_hi_ratio)

    return


def register_h2_masses(self, catalogue, aperture_sizes):

    # Loop over aperture sizes
    for aperture_size in aperture_sizes:

        # Fetch H2 mass
        H2_mass = catalogue.get_quantity(
            f"gas_hydrogen_species_masses.H2_mass_{aperture_size}_kpc"
        )

        # Label of the derived field
        H2_mass.name = f"$M_{{\\rm H_2}}$ ({aperture_size} kpc)"

        # Compute H2 mass with correction due to He
        He_mass = catalogue.get_quantity(
            f"gas_H_and_He_masses.He_mass_{aperture_size}_kpc"
        )
        H_mass = catalogue.get_quantity(
            f"gas_H_and_He_masses.H_mass_{aperture_size}_kpc"
        )

        H2_mass_with_He = unyt.unyt_array(np.zeros(H2_mass.shape), units=H2_mass.units)
        H2_mass_with_He[H_mass > 0.0] = H2_mass[H_mass > 0.0] * (
            1.0 + He_mass[H_mass > 0.0] / H_mass[H_mass > 0.0]
        )

        # Add label
        H2_mass_with_He.name = f"$M_{{\\rm H_2}}$ (incl. He, {aperture_size} kpc)"

        # Register field
        setattr(self, f"gas_H2_mass_{aperture_size}_kpc", H2_mass)
        setattr(self, f"gas_H2_plus_He_mass_{aperture_size}_kpc", H2_mass_with_He)

    return


def register_cold_gas_mass_ratios(self, catalogue, aperture_sizes):

    # Loop over aperture sizes
    for aperture_size in aperture_sizes:

        # Fetch HI and H2 masses in apertures of a given size
        HI_mass = catalogue.get_quantity(
            f"gas_hydrogen_species_masses.HI_mass_{aperture_size}_kpc"
        )
        H2_mass = catalogue.get_quantity(
            f"gas_hydrogen_species_masses.H2_mass_{aperture_size}_kpc"
        )

        # Compute neutral H mass (HI + H2)
        neutral_H_mass = HI_mass + H2_mass
        # Add label
        neutral_H_mass.name = f"$M_{{\\rm HI + H_2}}$ ({aperture_size} kpc)"

        # Fetch total stellar mass
        stellar_mass = catalogue.get_quantity(
            f"apertures.mass_star_{aperture_size}_kpc"
        )
        # Fetch mass of star-forming gas
        sf_mass = catalogue.get_quantity(f"apertures.mass_gas_sf_{aperture_size}_kpc")

        # Compute neutral H mass to stellar mass ratio
        neutral_H_to_stellar_fraction = unyt.unyt_array(
            np.zeros(neutral_H_mass.shape), units=unyt.dimensionless
        )
        neutral_H_to_stellar_fraction[stellar_mass > 0.0] = (
            neutral_H_mass[stellar_mass > 0.0] / stellar_mass[stellar_mass > 0.0]
        )
        neutral_H_to_stellar_fraction.name = (
            f"$M_{{\\rm HI + H_2}} / M_*$ ({aperture_size} kpc)"
        )

        # Compute molecular H mass to molecular plus stellar mass fraction
        molecular_H_to_molecular_plus_stellar_fraction = unyt.unyt_array(
            np.zeros(H2_mass.shape), units=unyt.dimensionless
        )
        mask_positive = (H2_mass + stellar_mass) > 0.0
        molecular_H_to_molecular_plus_stellar_fraction[mask_positive] = H2_mass[
            mask_positive
        ] / (H2_mass[mask_positive] + stellar_mass[mask_positive])
        molecular_H_to_molecular_plus_stellar_fraction.name = (
            f"$M_{{\\rm H_2}} / (M_* + M_{{\\rm H_2}})$ ({aperture_size} kpc)"
        )

        # Compute molecular H mass to neutral H mass ratio
        molecular_H_to_neutral_fraction = unyt.unyt_array(
            np.zeros(H2_mass.shape), units=unyt.dimensionless
        )
        molecular_H_to_neutral_fraction[neutral_H_mass > 0.0] = (
            H2_mass[neutral_H_mass > 0.0] / neutral_H_mass[neutral_H_mass > 0.0]
        )
        molecular_H_to_neutral_fraction.name = (
            f"$M_{{\\rm H_2}} / M_{{\\rm HI + H_2}}$ ({aperture_size} kpc)"
        )

        # Compute neutral H mass to baryonic mass fraction
        neutral_H_to_baryonic_fraction = unyt.unyt_array(
            np.zeros(neutral_H_mass.shape), units=unyt.dimensionless
        )
        mask_positive = (neutral_H_mass + stellar_mass) > 0.0
        neutral_H_to_baryonic_fraction[mask_positive] = neutral_H_mass[
            mask_positive
        ] / (neutral_H_mass[mask_positive] + stellar_mass[mask_positive])
        neutral_H_to_baryonic_fraction.name = (
            f"$M_{{\\rm HI + H_2}}/((M_*+ M_{{\\rm HI + H_2}})$ ({aperture_size} kpc)"
        )

        # Compute HI mass to neutral H mass ratio
        HI_to_neutral_H_fraction = unyt.unyt_array(
            np.zeros(HI_mass.shape), units=unyt.dimensionless
        )
        HI_to_neutral_H_fraction[neutral_H_mass > 0.0] = (
            HI_mass[neutral_H_mass > 0.0] / neutral_H_mass[neutral_H_mass > 0.0]
        )
        HI_to_neutral_H_fraction.name = (
            f"$M_{{\\rm HI}}/M_{{\\rm HI + H_2}}$ ({aperture_size} kpc)"
        )

        # Compute H2 mass to neutral H mass ratio
        H2_to_neutral_H_fraction = unyt.unyt_array(
            np.zeros(H2_mass.shape), units=unyt.dimensionless
        )
        H2_to_neutral_H_fraction[neutral_H_mass > 0.0] = (
            H2_mass[neutral_H_mass > 0.0] / neutral_H_mass[neutral_H_mass > 0.0]
        )
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
        setattr(self, f"gas_HI_to_sf_fraction_{aperture_size}_kpc", HI_to_sf_fraction)
        setattr(self, f"gas_H2_to_sf_fraction_{aperture_size}_kpc", H2_to_sf_fraction)
        setattr(
            self,
            f"gas_sf_to_stellar_fraction_{aperture_size}_kpc",
            sf_to_stellar_fraction,
        )
        setattr(self, f"has_neutral_gas_{aperture_size}_kpc", neutral_H_mass > 0.0)

    return


def register_species_fractions(self, catalogue, aperture_sizes):

    # Loop over aperture sizes
    for aperture_size in aperture_sizes:

        # Compute galaxy area (pi r^2)
        gal_area = (
            2
            * np.pi
            * catalogue.get_quantity(
                f"projected_apertures.projected_1_rhalfmass_star_{aperture_size}_kpc"
            )
            ** 2
        )

        # Stellar mass
        M_star = catalogue.get_quantity(f"apertures.mass_star_{aperture_size}_kpc")
        M_star_projected = catalogue.get_quantity(
            f"projected_apertures.projected_1_mass_star_{aperture_size}_kpc"
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
        mu_star = unyt.unyt_array(
            np.zeros(M_star_projected.shape),
            units=M_star_projected.units / gal_area.units,
        )
        mu_star[gal_area > 0.0] = (
            M_star_projected[gal_area > 0.0] / gal_area[gal_area > 0.0]
        )
        mu_star.name = f"$M_{{*, {aperture_size} {{\\rm kpc}}}} / \\pi R_{{*, {aperture_size} {{\\rm kpc}}}}^2$"

        # Atomic hydrogen mass in apertures
        HI_mass = catalogue.get_quantity(
            f"gas_hydrogen_species_masses.HI_mass_{aperture_size}_kpc"
        )
        HI_mass.name = f"HI Mass ({aperture_size} kpc)"

        # Molecular hydrogen mass in apertures
        H2_mass = catalogue.get_quantity(
            f"gas_hydrogen_species_masses.H2_mass_{aperture_size}_kpc"
        )
        H2_mass.name = f"H$_2$ Mass ({aperture_size} kpc)"

        # Compute H2 mass with correction due to He
        He_mass = catalogue.get_quantity(
            f"gas_H_and_He_masses.He_mass_{aperture_size}_kpc"
        )
        H_mass = catalogue.get_quantity(
            f"gas_H_and_He_masses.H_mass_{aperture_size}_kpc"
        )

        H2_mass_with_He = unyt.unyt_array(np.zeros(H2_mass.shape), units=H2_mass.units)
        H2_mass_with_He[H_mass > 0.0] = H2_mass[H_mass > 0.0] * (
            1.0 + He_mass[H_mass > 0.0] / H_mass[H_mass > 0.0]
        )
        H2_mass_with_He.name = f"$M_{{\\rm H_2}}$ (incl. He, {aperture_size} kpc)"

        # Atomic hydrogen to stellar mass in apertures
        hi_to_stellar_mass = unyt.unyt_array(
            np.zeros(HI_mass.shape), units=unyt.dimensionless
        )
        hi_to_stellar_mass[M_star > 0.0] = HI_mass[M_star > 0.0] / M_star[M_star > 0.0]
        hi_to_stellar_mass.name = f"$M_{{\\rm HI}} / M_*$ ({aperture_size} kpc)"

        # Molecular hydrogen mass to stellar mass in apertures
        h2_to_stellar_mass = unyt.unyt_array(
            np.zeros(H2_mass.shape), units=unyt.dimensionless
        )
        h2_to_stellar_mass[M_star > 0.0] = H2_mass[M_star > 0.0] / M_star[M_star > 0.0]
        h2_to_stellar_mass.name = f"$M_{{\\rm H_2}} / M_*$ ({aperture_size} kpc)"

        # Molecular hydrogen mass to stellar mass in apertures
        h2_plus_he_to_stellar_mass = unyt.unyt_array(
            np.zeros(H2_mass_with_He.shape), units=unyt.dimensionless
        )
        h2_plus_he_to_stellar_mass[M_star > 0.0] = (
            H2_mass_with_He[M_star > 0.0] / M_star[M_star > 0.0]
        )
        h2_plus_he_to_stellar_mass.name = (
            f"$M_{{\\rm H_2}} / M_*$ (incl. He, {aperture_size} kpc)"
        )

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
            f"h2_plus_he_to_stellar_mass_{aperture_size}_kpc",
            h2_plus_he_to_stellar_mass,
        )
        setattr(
            self,
            f"neutral_to_stellar_mass_{aperture_size}_kpc",
            neutral_to_stellar_mass,
        )

    return


def register_gas_fraction(self, catalogue):

    Omega_m = catalogue.units.cosmology.Om0
    Omega_b = catalogue.units.cosmology.Ob0

    M_500 = catalogue.get_quantity("spherical_overdensities.mass_500_rhocrit")
    M_500_gas = catalogue.get_quantity("spherical_overdensities.mass_gas_500_rhocrit")
    M_500_star = catalogue.get_quantity("spherical_overdensities.mass_star_500_rhocrit")
    M_500_baryon = M_500_gas + M_500_star

    f_b_500 = unyt.unyt_array(np.zeros(M_500.shape), units=unyt.dimensionless)
    f_b_500[M_500 > 0.0] = (M_500_baryon[M_500 > 0.0] / M_500[M_500 > 0.0]) / (
        Omega_b / Omega_m
    )
    name = "$f_{\\rm b, 500, true} / (\\Omega_{\\rm b} / \\Omega_{\\rm m})$"
    f_b_500.name = name

    f_gas_500 = unyt.unyt_array(np.zeros(M_500.shape), units=unyt.dimensionless)
    f_gas_500[M_500 > 0.0] = (M_500_gas[M_500 > 0.0] / M_500[M_500 > 0.0]) / (
        Omega_b / Omega_m
    )
    name = "$f_{\\rm gas, 500, true} / (\\Omega_{\\rm b} / \\Omega_{\\rm m})$"
    f_gas_500.name = name

    f_star_500 = unyt.unyt_array(np.zeros(M_500.shape), units=unyt.dimensionless)
    f_star_500[M_500 > 0.0] = (M_500_star[M_500 > 0.0] / M_500[M_500 > 0.0]) / (
        Omega_b / Omega_m
    )
    name = "$f_{\\rm star, 500, true} / (\\Omega_{\\rm b} / \\Omega_{\\rm m})$"
    f_star_500.name = name

    setattr(self, "baryon_fraction_true_R500", f_b_500)
    setattr(self, "gas_fraction_true_R500", f_gas_500)
    setattr(self, "star_fraction_true_R500", f_star_500)

    return


def register_los_star_veldisp(self, catalogue):
    for aperture_size in [10, 30]:
        veldisp = catalogue.get_quantity(f"apertures.veldisp_star_{aperture_size}_kpc")
        los_veldisp = veldisp / np.sqrt(3.0)
        los_veldisp.name = f"LOS stellar velocity dispersion ({aperture_size} kpc)"
        setattr(self, f"los_veldisp_star_{aperture_size}_kpc", los_veldisp)

    return


def register_stellar_mass_scatter(self, catalogue, stellar_mass_scatter_amplitude):

    stellar_mass = catalogue.get_quantity("apertures.mass_star_50_kpc")
    stellar_mass_with_scatter = unyt.unyt_array(
        np.zeros(stellar_mass.shape), units=stellar_mass.units
    )
    stellar_mass_with_scatter[stellar_mass > 0.0] = np.random.lognormal(
        np.log(stellar_mass[stellar_mass > 0.0].value),
        stellar_mass_scatter_amplitude * np.log(10.0),
    )
    stellar_mass_with_scatter.name = f"Stellar mass $M_*$ with {stellar_mass_scatter_amplitude:.1f} dex scatter (50 kpc)"
    setattr(self, f"mass_star_with_scatter_50_kpc", stellar_mass_with_scatter)

    return


def register_SNIa_rates(self, catalogue, aperture_sizes):

    # Loop over apertures
    for aperture_size in aperture_sizes:

        # Get stellar mass and SNIa rate
        stellar_mass = catalogue.get_quantity(
            f"apertures.mass_star_{aperture_size}_kpc"
        )
        SNIa_rate = catalogue.get_quantity(f"snia_rates.snia_rates_{aperture_size}_kpc")

        # calculate the SNIa rate per stellar mass
        SNIa_rate_per_stellar_mass = unyt.unyt_array(
            np.zeros(SNIa_rate.shape), units=SNIa_rate.units / stellar_mass.units
        )
        SNIa_rate_per_stellar_mass[stellar_mass > 0.0] = (
            SNIa_rate[stellar_mass > 0.0] / stellar_mass[stellar_mass > 0.0]
        )

        # Name (label) of the derived field
        SNIa_rate_per_stellar_mass.name = (
            f"SNIa rate / $M_\\star$ ({aperture_size} kpc)"
        )
        setattr(
            self,
            f"snia_rate_per_stellar_mass_{aperture_size}_kpc",
            SNIa_rate_per_stellar_mass,
        )

    return


# Register derived fields
register_SNIa_rates(self, catalogue, aperture_sizes_30_50_100_kpc)
register_specific_star_formation_rates(self, catalogue, aperture_sizes_30_50_100_kpc)
register_star_metallicities(
    self, catalogue, aperture_sizes_30_50_100_kpc, solar_metal_mass_fraction
)
register_stellar_to_halo_mass_ratios(self, catalogue, aperture_sizes_30_50_100_kpc)
register_projected_stellar_masses(self, catalogue, aperture_sizes_10_30_50_100_kpc)
register_oxygen_to_hydrogen(self, catalogue, aperture_sizes_30_50_100_kpc)
register_nitrogen_to_oxygen(self, catalogue, aperture_sizes_30_50_100_kpc)
register_carbon_to_oxygen(self, catalogue, aperture_sizes_30_50_100_kpc)
register_cold_dense_gas_metallicity(
    self,
    catalogue,
    aperture_sizes_30_50_100_kpc,
    solar_metal_mass_fraction,
    twelve_plus_log_OH_solar,
)
register_iron_to_hydrogen(
    self, catalogue, aperture_sizes_30_50_100_kpc, solar_fe_abundance
)
register_hi_masses(self, catalogue, aperture_sizes_30_50_100_kpc)
register_h2_masses(self, catalogue, aperture_sizes_30_50_100_kpc)
register_dust_to_hi_ratio(self, catalogue, aperture_sizes_30_50_100_kpc)
register_cold_gas_mass_ratios(self, catalogue, aperture_sizes_30_50_100_kpc)
register_species_fractions(self, catalogue, aperture_sizes_30_50_100_kpc)
register_los_star_veldisp(self, catalogue)
register_star_Mg_and_O_to_Fe(self, catalogue, aperture_sizes_30_50_100_kpc)
register_gas_fraction(self, catalogue)
register_stellar_mass_scatter(self, catalogue, stellar_mass_scatter_amplitude)
register_dust(
    self,
    catalogue,
    aperture_sizes_30_50_100_kpc,
    solar_metal_mass_fraction,
    twelve_plus_log_OH_solar,
)
register_star_magnitudes(self, catalogue, aperture_sizes_30_50_100_kpc)

register_corrected_star_magnitudes(self, catalogue, aperture_sizes_30_50_100_kpc)

# register_corrected_star_magnitudes(self, catalogue, aperture_sizes_30_50_100_kpc, add_dust = False)

register_chabrier_masses(self, catalogue, aperture_sizes_30_50_100_kpc)
