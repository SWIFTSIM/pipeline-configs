"""
This special mode overides all stellar masses to include Eddington bias.
The bias is added using the description by Behroozi (2019)
"""

aperture_sizes = [30, 50, 100]

for aperture_size in aperture_sizes:
    stellar_mass = getattr(catalogue.apertures, f"mass_star_{aperture_size}_kpc")
    bias_std = np.min(np.array([0.07 + 0.071 * catalogue.z, 0.3]))
    bias_factors = 10 ** (np.random.normal(0, bias_std, len(stellar_mass)))

    stellar_mass_with_bias = unyt.unyt_array(stellar_mass * bias_factors)
    stellar_mass_with_bias.name = f"Stellar Mass $M_*$ ({aperture_size} kpc)"

    setattr(
        catalogue.apertures, f"mass_star_{aperture_size}_kpc", stellar_mass_with_bias
    )
