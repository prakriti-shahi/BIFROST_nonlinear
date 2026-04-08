"""
Spontaneous Brillouin scattering noise model for optical fibers.

Computes the Brillouin gain spectrum, SBS threshold, and backward-propagating
spontaneous Brillouin noise relevant to WDM quantum-classical coexistence on
a shared fiber.  The model covers backward SBS only; forward Brillouin (GAWBS)
is ~30 dB weaker and is not included.

Key physics
-----------
Brillouin scattering couples a pump photon to a backward-propagating Stokes
photon and a longitudinal acoustic phonon.  Phase matching requires

    nu_B = 2 n_eff v_A / lambda_pump   (~11 GHz at 1550 nm in SiO2)

The gain spectrum is Lorentzian with FWHM ~ 20-50 MHz (set by phonon lifetime).
At room temperature the phonon occupation n_th ~ 560 >> 1, so spontaneous
emission is intense within the narrow Brillouin line.

References
----------
[1] Agrawal, G.P., Nonlinear Fiber Optics, 6th ed. (Academic, 2019), Ch. 9.
[2] Smith, R.G., Appl. Opt. 11, 2489 (1972).
[3] Nikles, M. et al., J. Lightwave Technol. 15, 1842 (1997).
[4] Kobyakov, A. et al., Adv. Opt. Photon. 2, 1 (2010).
[5] Tkach, R.W. et al., J. Lightwave Technol. 5, 1380 (1987).
[6] Boyd, R.W., Nonlinear Optics, 4th ed. (Academic, 2020), Ch. 9.
"""

import warnings
import numpy as np
from scipy.integrate import simpson

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_HBAR = 1.054571817e-34   # J s
_KB   = 1.380649e-23      # J / K
_C    = 299792458.0        # m / s
_PI   = np.pi

# ---------------------------------------------------------------------------
# Material parameters  (germanosilicate single-mode fibers)
# ---------------------------------------------------------------------------

BRIL_VA_SIO2   = 5970.0    # m/s  — longitudinal acoustic velocity, pure SiO2 [1,3]
BRIL_VA_COEFF  = -0.274    # fractional v_A change per unit GeO2 mole fraction [3,5]

BRIL_G_PEAK    = 5.0e-11   # m/W  — peak gain, pure SiO2 [1,4]
BRIL_G_GEO2_K  = 4.4       # decay constant for g_B vs GeO2 (see g_B_peak_GeO2)
BRIL_G_GEO2_XMAX = 0.08   # validity limit of the empirical g_B(x) fit

BRIL_DFREQ_HZ  = 25.0e6   # Hz  — FWHM Lorentzian linewidth, SMF-28, 300 K [3,4]
BRIL_GAMMA_RAD = 2 * _PI * BRIL_DFREQ_HZ

BRIL_G_THRESH  = 21.0      # Smith (1972) threshold gain factor [2]

# ---------------------------------------------------------------------------
# Fiber loss  (empirical SMF-28 model, duplicated from raman.py)
# ---------------------------------------------------------------------------

_LOSS_A_RAY = 0.78    # dB/km um^4  (Rayleigh)
_LOSS_FLOOR = 0.065   # dB/km       (IR absorption)


def _fiber_loss_m(wavelength_m):
    """Power attenuation coefficient alpha (1/m) for SMF-28."""
    lam_um = np.asarray(wavelength_m, dtype=float) * 1e6
    dB_km  = _LOSS_A_RAY / lam_um**4 + _LOSS_FLOOR
    return dB_km / (10.0 * np.log10(np.e)) / 1e3


# ═══════════════════════════════════════════════════════════════════════════
# Brillouin frequency shift
# ═══════════════════════════════════════════════════════════════════════════

def brillouin_freq_shift(lambda_pump, n_eff=1.4447, v_acoustic=BRIL_VA_SIO2,
                         m_GeO2=0.0):
    r"""Brillouin frequency shift for backward SBS.

    .. math::
        \nu_B = \frac{2\, n_\mathrm{eff}\, v_A}{\lambda_\mathrm{pump}}

    Parameters
    ----------
    lambda_pump : float or array_like
        Pump wavelength in vacuum (m).
    n_eff : float
        Effective index of LP01 mode.  Default 1.4447 (SMF-28, 1550 nm).
    v_acoustic : float
        Longitudinal acoustic velocity in pure SiO2 (m/s).
    m_GeO2 : float
        GeO2 mole fraction.  Corrects velocity via
        ``v_A = v_acoustic * (1 + BRIL_VA_COEFF * m_GeO2)``.

    Returns
    -------
    float or ndarray
        Brillouin shift nu_B (Hz).
    """
    v_A = v_acoustic * (1.0 + BRIL_VA_COEFF * m_GeO2)
    return 2.0 * n_eff * v_A / np.asarray(lambda_pump, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════
# Peak gain vs GeO2 concentration
# ═══════════════════════════════════════════════════════════════════════════

def g_B_peak_GeO2(m_GeO2=0.0, g_B_SiO2=BRIL_G_PEAK, k=BRIL_G_GEO2_K):
    r"""Peak Brillouin gain coefficient as a function of GeO2 doping.

    Empirical fit to Nikles et al. (1997) Table II [3]:

    .. math::
        g_B(x) = g_B^\mathrm{SiO_2} \exp\!\bigl(-k\,\sqrt{x}\bigr)

    The dominant mechanism is the reduction of the photoelastic coefficient
    p_12 with increasing GeO2, which weakens the acousto-optic coupling
    faster than the density and refractive-index increases enhance it.

    **Validity**: germanosilicate fibers with 0 <= x <= 0.08 (8 mol%).
    Above 8 mol%, acousto-optic overlap effects that depend on the specific
    waveguide geometry become significant and this bulk-fit breaks down.
    A warning is issued and the pure-SiO2 constant ``BRIL_G_PEAK`` is
    returned for any x > ``BRIL_G_GEO2_XMAX``.

    Calibration (Nikles 1997, 1320 nm pump):

    =======  ===========  ===========  ==========
    x        Fiber        Expt (m/W)   Model (m/W)
    =======  ===========  ===========  ==========
    0.0000   pure SiO2    5.00e-11     5.00e-11
    0.0365   SMF-28       2.16e-11     2.16e-11
    0.0750   DSF          1.53e-11     1.50e-11
    =======  ===========  ===========  ==========

    Parameters
    ----------
    m_GeO2 : float
        GeO2 mole fraction (0 = pure SiO2).
    g_B_SiO2 : float
        Peak gain for pure SiO2 (m/W).
    k : float
        Empirical decay constant.

    Returns
    -------
    float
        Peak Brillouin gain g_B (m/W).
    """
    x = float(m_GeO2)
    if x > BRIL_G_GEO2_XMAX:
        warnings.warn(
            f"g_B_peak_GeO2: GeO2 mole fraction {x:.3f} exceeds validated "
            f"range [0, {BRIL_G_GEO2_XMAX}].  Returning BRIL_G_PEAK "
            f"({BRIL_G_PEAK:.1e} m/W).  Supply a measured g_B instead.",
            UserWarning, stacklevel=2,
        )
        return g_B_SiO2
    return g_B_SiO2 * np.exp(-k * np.sqrt(abs(x)))


# ═══════════════════════════════════════════════════════════════════════════
# Gain spectrum (Lorentzian)
# ═══════════════════════════════════════════════════════════════════════════

def g_B_lorentzian(Omega, Omega_B, g_B_peak=BRIL_G_PEAK,
                   Gamma_B=BRIL_GAMMA_RAD):
    r"""Lorentzian Brillouin gain spectrum.

    .. math::
        g_B(\Omega) = g_B^\mathrm{peak}
            \frac{(\Gamma_B/2)^2}{(\Omega - \Omega_B)^2 + (\Gamma_B/2)^2}

    Parameters
    ----------
    Omega : array_like
        Angular frequency offset from pump (rad/s).
    Omega_B : float
        Brillouin angular shift 2 pi nu_B (rad/s).
    g_B_peak : float
        Peak gain at resonance (m/W).
    Gamma_B : float
        FWHM linewidth (rad/s).

    Returns
    -------
    ndarray
        g_B(Omega) in m/W.
    """
    Omega = np.asarray(Omega, dtype=float)
    hwhm  = Gamma_B / 2.0
    return g_B_peak * hwhm**2 / ((Omega - Omega_B)**2 + hwhm**2)


# ═══════════════════════════════════════════════════════════════════════════
# Effective interaction length (backward geometry)
# ═══════════════════════════════════════════════════════════════════════════

def effective_length_backward(L, alpha_pump, alpha_signal=None):
    r"""Effective length for backward Brillouin interaction.

    .. math::
        L_\mathrm{eff}^\mathrm{back} =
        \frac{1 - e^{-(\alpha_p + \alpha_s) L}}{\alpha_p + \alpha_s}

    Long-fiber asymptote: 1/(alpha_p + alpha_s) ~ 10.9 km at 1550 nm.

    Parameters
    ----------
    L : float
        Physical fiber length (m).
    alpha_pump : float or array_like
        Pump loss coefficient (1/m).
    alpha_signal : float or array_like, optional
        Stokes loss coefficient (1/m).  Defaults to ``alpha_pump``.

    Returns
    -------
    ndarray
        Backward effective length (m).
    """
    alpha_pump = np.asarray(alpha_pump, dtype=float)
    if alpha_signal is None:
        alpha_signal = alpha_pump
    else:
        alpha_signal = np.asarray(alpha_signal, dtype=float)

    alpha_sum = alpha_pump + alpha_signal
    safe = np.where(alpha_sum > 1e-12, alpha_sum, 1.0)
    Leff = -np.expm1(-safe * L) / safe
    return np.where(alpha_sum > 1e-12, Leff, float(L))


# ═══════════════════════════════════════════════════════════════════════════
# Thermal phonon occupancy
# ═══════════════════════════════════════════════════════════════════════════

def thermal_phonon_number(Omega, T_K):
    r"""Bose-Einstein occupancy of acoustic phonons.

    At ~11 GHz and 300 K:  n_th ~ kT / hbar Omega ~ 560.

    Parameters
    ----------
    Omega : array_like
        Phonon angular frequency (rad/s).
    T_K : float
        Temperature (K).

    Returns
    -------
    ndarray
        Thermal phonon number n_th.
    """
    Omega = np.asarray(np.abs(Omega), dtype=float)
    x = _HBAR * Omega / (_KB * T_K)
    return np.where(x > 500.0, 0.0, 1.0 / np.expm1(x))


# ═══════════════════════════════════════════════════════════════════════════
# SBS threshold
# ═══════════════════════════════════════════════════════════════════════════

def brillouin_threshold(A_eff, L, alpha=None, lambda_pump=1550e-9,
                        g_B_peak=BRIL_G_PEAK, G_th=BRIL_G_THRESH):
    r"""SBS threshold power via the Smith (1972) criterion [2].

    .. math::
        P_\mathrm{th} = \frac{G_\mathrm{th}\, A_\mathrm{eff}}
                             {g_B\, L_\mathrm{eff}^\mathrm{back}}

    Parameters
    ----------
    A_eff : float
        Effective mode area (m^2).
    L : float
        Fiber length (m).
    alpha : float, optional
        Loss coefficient (1/m).  ``None`` uses the SMF-28 model.
    lambda_pump : float
        Pump wavelength (m).
    g_B_peak : float
        Peak Brillouin gain (m/W).
    G_th : float
        Threshold gain factor (default 21).

    Returns
    -------
    dict
        Keys: ``P_threshold_W``, ``P_threshold_mW``, ``G_B_at_threshold``,
        ``L_eff_back_m``, ``g_B_peak``, ``alpha_m``, ``formula``.
    """
    if alpha is None:
        alpha = float(_fiber_loss_m(lambda_pump))

    Leff = float(effective_length_backward(L, alpha, alpha))
    P_th = G_th * A_eff / (g_B_peak * Leff)

    return {
        'P_threshold_W':    P_th,
        'P_threshold_mW':   P_th * 1e3,
        'G_B_at_threshold': G_th,
        'L_eff_back_m':     Leff,
        'g_B_peak':         g_B_peak,
        'alpha_m':          alpha,
        'formula': (f"P_th = {G_th} * {A_eff*1e12:.1f} um^2 / "
                    f"({g_B_peak:.1e} m/W * {Leff/1e3:.2f} km) "
                    f"= {P_th*1e3:.2f} mW"),
    }


def check_sbs_threshold(P_pump, A_eff, L, alpha=None, lambda_pump=1550e-9,
                        g_B_peak=BRIL_G_PEAK, threshold_fraction=0.80):
    """Check whether pump power is near or above the SBS threshold.

    Issues a ``UserWarning`` if ``P_pump > threshold_fraction * P_th``.

    Parameters
    ----------
    P_pump : float
        Pump power at fiber input (W).
    A_eff, L, alpha, lambda_pump, g_B_peak
        Forwarded to :func:`brillouin_threshold`.
    threshold_fraction : float
        Warning fraction of P_th (default 0.80).

    Returns
    -------
    dict
        Keys: ``P_threshold_W``, ``P_pump_W``, ``fraction_of_threshold``,
        ``sbs_gain_parameter``, ``above_threshold``, ``valid``,
        ``recommendation``.
    """
    thresh = brillouin_threshold(A_eff, L, alpha, lambda_pump, g_B_peak)
    P_th   = thresh['P_threshold_W']
    G_B    = g_B_peak * P_pump * thresh['L_eff_back_m'] / A_eff
    frac   = P_pump / P_th
    above  = frac > 1.0
    valid  = frac < threshold_fraction

    if not valid:
        warnings.warn(
            f"Pump power ({P_pump*1e3:.2f} mW) is {frac*100:.0f}% of SBS "
            f"threshold ({P_th*1e3:.2f} mW).  Consider SBS suppression.",
            UserWarning, stacklevel=2,
        )

    tag = ("ABOVE threshold." if above
           else "Safe." if valid
           else "Approaching threshold.")

    return {
        'P_threshold_W':         P_th,
        'P_pump_W':              P_pump,
        'fraction_of_threshold': frac,
        'sbs_gain_parameter':    G_B,
        'above_threshold':       above,
        'valid':                 valid,
        'recommendation':        f"P_pump = {P_pump*1e3:.2f} mW — {tag}",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Spontaneous Brillouin photon-rate spectral density
# ═══════════════════════════════════════════════════════════════════════════

def spbs_photon_rate_density(Omega, lambda_pump, P_pump, L, A_eff, T_K,
                             n_eff=1.4447, v_acoustic=BRIL_VA_SIO2,
                             m_GeO2=0.0, g_B_peak=BRIL_G_PEAK,
                             Gamma_B=BRIL_GAMMA_RAD,
                             alpha_pump=None, alpha_signal=None):
    r"""Spectral density of backward spontaneous Brillouin photons.

    .. math::
        \frac{dN}{d\Omega\,dt} =
        \frac{1}{2\pi}\,\frac{g_B(\Omega)}{A_\mathrm{eff}}\,
        P_\mathrm{pump}\,L_\mathrm{eff}^\mathrm{back}\,n_\mathrm{th}

    The returned density is in units of photons / (s · rad/s) and describes
    backward-propagating noise exiting the fiber at z = 0.

    Parameters
    ----------
    Omega : array_like
        Angular frequency offset from pump (rad/s).
    lambda_pump : float
        Pump wavelength (m).
    P_pump : float
        Pump power at fiber input (W).
    L : float
        Fiber length (m).
    A_eff : float
        Effective mode area (m^2).
    T_K : float
        Temperature (K).
    n_eff, v_acoustic, m_GeO2, g_B_peak, Gamma_B
        Brillouin parameters (see individual defaults).
    alpha_pump, alpha_signal : float, optional
        Loss coefficients (1/m).  ``None`` uses the SMF-28 model.

    Returns
    -------
    ndarray
        dN/dOmega/dt  (photons s^-1 (rad/s)^-1).
    """
    Omega = np.asarray(Omega, dtype=float)

    nu_B    = brillouin_freq_shift(lambda_pump, n_eff, v_acoustic, m_GeO2)
    Omega_B = 2.0 * _PI * nu_B

    if alpha_pump is None:
        alpha_pump = float(_fiber_loss_m(lambda_pump))
    if alpha_signal is None:
        alpha_signal = alpha_pump

    Leff = effective_length_backward(L, alpha_pump, alpha_signal)
    gB   = g_B_lorentzian(Omega, Omega_B, g_B_peak, Gamma_B)
    nth  = thermal_phonon_number(Omega, T_K)

    return (gB / A_eff) * P_pump * Leff * nth / (2.0 * _PI)


# Legacy alias (matches earlier API)
spBril_photon_rate_density = spbs_photon_rate_density


def spbs_noise_in_channel(lambda_pump, lambda_channel, delta_lambda,
                          P_pump, L, A_eff, T_K, N_points=501,
                          n_eff=1.4447, v_acoustic=BRIL_VA_SIO2,
                          m_GeO2=0.0, g_B_peak=BRIL_G_PEAK,
                          Gamma_B=BRIL_GAMMA_RAD,
                          alpha_pump=None, alpha_signal=None):
    """Total backward spontaneous Brillouin noise in a channel bandwidth.

    Integrates :func:`spbs_photon_rate_density` over a rectangular filter of
    width ``delta_lambda`` centred at ``lambda_channel``.

    Parameters
    ----------
    lambda_pump : float
        Pump wavelength (m).
    lambda_channel : float
        Channel centre wavelength (m).
    delta_lambda : float
        Channel filter bandwidth (m, full-width).
    P_pump : float
        Pump power at fiber input (W).
    L, A_eff, T_K
        Fiber length (m), mode area (m^2), temperature (K).
    N_points : int
        Quadrature points (odd).
    n_eff, v_acoustic, m_GeO2, g_B_peak, Gamma_B,
    alpha_pump, alpha_signal
        Forwarded to :func:`spbs_photon_rate_density`.

    Returns
    -------
    dict
        Keys: ``backward_photon_rate``, ``backward_power_W``,
        ``nu_B_Hz``, ``lambda_Stokes_nm``, ``channel_is_near_Brillouin``,
        ``P_threshold_mW``, ``pump_fraction_of_threshold``.
    """
    if N_points % 2 == 0:
        N_points += 1

    omega_pump = 2.0 * _PI * _C / lambda_pump

    nu_B    = float(brillouin_freq_shift(lambda_pump, n_eff, v_acoustic, m_GeO2))
    Omega_B = 2.0 * _PI * nu_B
    lam_S   = _C / (_C / lambda_pump - nu_B)

    # Channel overlap check
    nu_offset = abs(_C / lambda_channel - (_C / lambda_pump - nu_B))
    near      = nu_offset < 10.0 * BRIL_DFREQ_HZ

    # Wavelength grid over the channel filter
    lam_arr = np.linspace(lambda_channel - delta_lambda / 2.0,
                          lambda_channel + delta_lambda / 2.0, N_points)
    om_arr    = 2.0 * _PI * _C / lam_arr
    Omega_arr = np.abs(omega_pump - om_arr)

    # Loss
    alpha_p = (float(alpha_pump) if alpha_pump is not None
               else float(_fiber_loss_m(lambda_pump)))
    alpha_s = (_fiber_loss_m(lam_arr) if alpha_signal is None
               else float(alpha_signal) * np.ones_like(lam_arr))

    Leff_arr = effective_length_backward(L, alpha_p, alpha_s)
    gB_arr   = g_B_lorentzian(Omega_arr, Omega_B, g_B_peak, Gamma_B)
    nth_arr  = thermal_phonon_number(Omega_arr, T_K)

    dNdOm = (gB_arr / A_eff) * P_pump * Leff_arr * nth_arr / (2.0 * _PI)

    # Integrate: dOmega = (2 pi c / lambda^2) dlambda
    rate = float(simpson(dNdOm * (2.0 * _PI * _C / lam_arr**2), x=lam_arr))
    omega_ch = 2.0 * _PI * _C / lambda_channel

    thresh = brillouin_threshold(A_eff, L, alpha_p, lambda_pump, g_B_peak)

    return {
        'backward_photon_rate':       rate,
        'backward_power_W':           rate * _HBAR * omega_ch,
        'Omega_B_rad':                Omega_B,
        'nu_B_Hz':                    nu_B,
        'delta_lambda_B_pm':          abs(lam_S - lambda_pump) * 1e12,
        'lambda_Stokes_nm':           lam_S * 1e9,
        'channel_is_near_Brillouin':  near,
        'P_threshold_mW':             thresh['P_threshold_mW'],
        'pump_fraction_of_threshold': P_pump / thresh['P_threshold_W'],
        'lambda_pump':                lambda_pump,
        'lambda_channel':             lambda_channel,
        'delta_lambda':               delta_lambda,
    }


# Legacy alias
spBril_noise_in_channel = spbs_noise_in_channel


# ═══════════════════════════════════════════════════════════════════════════
# Gain profile convenience function
# ═══════════════════════════════════════════════════════════════════════════

def brillouin_gain_profile(lambda_pump, freq_range_GHz=(-100, 100), N=4096,
                           n_eff=1.4447, v_acoustic=BRIL_VA_SIO2,
                           m_GeO2=0.0, g_B_peak=BRIL_G_PEAK,
                           Gamma_B=BRIL_GAMMA_RAD):
    """Compute the Brillouin gain profile over a frequency range.

    Parameters
    ----------
    lambda_pump : float
        Pump wavelength (m).
    freq_range_GHz : tuple
        (lo, hi) frequency offset from pump in GHz.
    N : int
        Number of points.
    n_eff, v_acoustic, m_GeO2, g_B_peak, Gamma_B
        Brillouin parameters.

    Returns
    -------
    dict
        Keys: ``freq_offset_GHz``, ``wavelength_m``, ``g_B_mW``,
        ``nu_B_GHz``, ``g_B_peak``, ``Gamma_B_MHz``.
    """
    nu_B    = float(brillouin_freq_shift(lambda_pump, n_eff, v_acoustic, m_GeO2))
    Omega_B = 2.0 * _PI * nu_B
    nu_pump = _C / lambda_pump

    nu_arr = np.linspace(nu_pump + freq_range_GHz[0] * 1e9,
                         nu_pump + freq_range_GHz[1] * 1e9, N)
    Om_arr = 2.0 * _PI * np.abs(nu_arr - nu_pump)
    gB_arr = g_B_lorentzian(Om_arr, Omega_B, g_B_peak, Gamma_B)

    return {
        'freq_offset_GHz': (nu_arr - nu_pump) / 1e9,
        'wavelength_m':    _C / nu_arr,
        'g_B_mW':          gB_arr,
        'nu_B_GHz':        nu_B / 1e9,
        'g_B_peak':        g_B_peak,
        'Gamma_B_MHz':     Gamma_B / (2.0 * _PI * 1e6),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Runtime attachment to FiberLength objects
# ═══════════════════════════════════════════════════════════════════════════

def install_brillouin_methods(fiber_obj):
    """Attach Brillouin noise methods to an existing ``FiberLength`` instance.

    After calling, the object gains ``calcSpBrilNoise``,
    ``brillouinThreshold``, and ``checkSBSThreshold`` methods.

    Parameters
    ----------
    fiber_obj : FiberLength
        Must expose ``.Aeff``, ``.Lt`` or ``.L0``, and ``.T0``.

    Returns
    -------
    FiberLength
        The same object, with methods attached.
    """
    import types

    def _calc(self, lambda_ref, lambda_quantum, delta_lambda, P_ref, **kw):
        T_K = getattr(self, 'T0', 22.0) + 273.15
        L   = getattr(self, 'Lt', getattr(self, 'L0', None))
        return spbs_noise_in_channel(
            lambda_ref, lambda_quantum, delta_lambda, P_ref,
            L, self.Aeff, T_K, **kw)

    def _thresh(self, lambda_ref=1550e-9):
        L = getattr(self, 'Lt', getattr(self, 'L0', None))
        return brillouin_threshold(self.Aeff, L, lambda_pump=lambda_ref)

    def _check(self, P_pump, lambda_ref=1550e-9, **kw):
        L = getattr(self, 'Lt', getattr(self, 'L0', None))
        return check_sbs_threshold(P_pump, self.Aeff, L,
                                   lambda_pump=lambda_ref, **kw)

    fiber_obj.calcSpBrilNoise    = types.MethodType(_calc,   fiber_obj)
    fiber_obj.brillouinThreshold = types.MethodType(_thresh, fiber_obj)
    fiber_obj.checkSBSThreshold  = types.MethodType(_check,  fiber_obj)
    return fiber_obj


# ═══════════════════════════════════════════════════════════════════════════
# Self-tests
# ═══════════════════════════════════════════════════════════════════════════

def _run_self_tests(verbose=True):
    """Run internal consistency checks.  Returns True if all pass."""
    results = []

    def check(name, condition, got='', expected=''):
        results.append(condition)
        if verbose:
            tag = '[PASS]' if condition else '[FAIL]'
            print(f'  {tag}  {name}')
            if not condition:
                print(f'         got {got}, expected {expected}')

    if verbose:
        print('=' * 58)
        print('brillouin.py — self-tests')
        print('=' * 58)

    nu_B    = brillouin_freq_shift(1550e-9)
    Omega_B = 2 * _PI * nu_B

    check('nu_B(1550 nm) in [10.5, 12.0] GHz',
          10.5e9 < nu_B < 12.0e9,
          f'{nu_B/1e9:.3f} GHz', '~11.1 GHz')

    check('g_B peaks at Omega_B',
          abs(float(g_B_lorentzian(Omega_B, Omega_B)) - BRIL_G_PEAK)
          / BRIL_G_PEAK < 1e-10)

    check('g_B HWHM condition',
          abs(float(g_B_lorentzian(Omega_B + BRIL_GAMMA_RAD/2, Omega_B))
              - BRIL_G_PEAK/2) / (BRIL_G_PEAK/2) < 1e-10)

    Om_int   = np.linspace(1e6, 10 * Omega_B, 1_000_000)
    integral = float(simpson(g_B_lorentzian(Om_int, Omega_B), x=Om_int))
    expected = BRIL_G_PEAK * _PI * BRIL_GAMMA_RAD / 2
    check('Lorentzian integral matches analytic',
          abs(integral - expected) / expected < 0.001,
          f'{integral:.4e}', f'{expected:.4e}')

    check('L_eff_back(alpha~0) = L',
          abs(float(effective_length_backward(25e3, 1e-30, 1e-30)) - 25e3)
          / 25e3 < 0.01)

    alpha = float(_fiber_loss_m(1550e-9))
    check('L_eff_back < L_eff_fwd',
          float(effective_length_backward(25e3, alpha))
          < (1 - np.exp(-alpha * 25e3)) / alpha)

    nth = float(thermal_phonon_number(Omega_B, 300.0))
    check('n_th(Omega_B, 300 K) in [500, 620]', 500 < nth < 620,
          f'{nth:.1f}', '~561')

    th = brillouin_threshold(85e-12, 25e3)
    check('P_th(25 km SMF-28) in [2, 6] mW',
          2.0 < th['P_threshold_mW'] < 6.0,
          f"{th['P_threshold_mW']:.2f} mW", '~3.6 mW')

    res = spbs_noise_in_channel(1310e-9, 1550e-9, 1e-9, 1e-3,
                                25e3, 85e-12, 300)
    check('SpBS far from Brillouin line ~ 0',
          res['backward_photon_rate'] < 1e-3)

    # g_B_peak_GeO2 calibration
    check('g_B_peak_GeO2(0) = BRIL_G_PEAK',
          abs(g_B_peak_GeO2(0.0) - BRIL_G_PEAK) < 1e-20)

    check('g_B_peak_GeO2(0.0365) ~ 2.16e-11 (N97 SMF-28)',
          abs(g_B_peak_GeO2(0.0365) - 2.16e-11) / 2.16e-11 < 0.01)

    check('g_B_peak_GeO2(0.075) ~ 1.50e-11 (N97 DSF)',
          abs(g_B_peak_GeO2(0.075) - 1.50e-11) / 1.50e-11 < 0.03)

    # Verify the >8% warning fires and returns BRIL_G_PEAK
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = g_B_peak_GeO2(0.15)
        check('g_B_peak_GeO2(0.15) warns and returns BRIL_G_PEAK',
              len(w) == 1 and val == BRIL_G_PEAK)

    ok = all(results)
    if verbose:
        print('=' * 58)
        print('ALL TESTS PASSED' if ok else 'SOME TESTS FAILED')
        print('=' * 58)
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    ok = _run_self_tests(verbose=True)
    if not ok:
        sys.exit(1)

    lam_p   = 1550e-9
    nu_B    = float(brillouin_freq_shift(lam_p))
    Omega_B = 2 * _PI * nu_B
    A_eff   = 85e-12

    print()
    print('Summary  (SMF-28, 1550 nm, 300 K)')
    print('-' * 42)
    print(f'  nu_B        = {nu_B/1e9:.3f} GHz')
    print(f'  Gamma_B     = {BRIL_DFREQ_HZ/1e6:.0f} MHz FWHM')
    print(f'  g_B (SiO2)  = {BRIL_G_PEAK:.2e} m/W')
    print(f'  g_B (SMF28) = {g_B_peak_GeO2(0.036):.2e} m/W')
    print(f'  n_th        = {thermal_phonon_number(Omega_B, 300.):.0f}')
    print()
    print(f'  {"L (km)":>8}  {"L_eff_back (km)":>16}  {"P_th (mW)":>10}')
    for Lkm in [5, 10, 25, 50, 100]:
        th = brillouin_threshold(A_eff, Lkm * 1e3, lambda_pump=lam_p)
        print(f'  {Lkm:>8}  {th["L_eff_back_m"]/1e3:>16.3f}'
              f'  {th["P_threshold_mW"]:>10.2f}')
