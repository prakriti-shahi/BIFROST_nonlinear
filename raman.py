"""
Spontaneous Raman scattering noise model for optical fibers.

Provides three Raman response models selectable via ``RAMAN_MODEL`` or a
per-call ``model=`` keyword:

``'bw'``
    Blow-Wood (1989) damped-oscillator model.  Analytic, fast.  Accurate
    near the 13 THz peak; overestimates the tail above ~20 THz.

``'tabulated'``
    Lin & Agrawal (2006) 50-point Kramers-Kronig-normalised spline.
    Accurate over 0-25 THz.  Requires ``raman_tabulated.py``.

``'hc'``
    Hollenbeck & Cantrell (2002) 13-mode intermediate-broadening model.
    Captures the 15 THz shoulder, multi-mode interference, and correct
    high-frequency tail.  Precomputed via sine transform at import time.

References
----------
[1] Blow, K.J. & Wood, D., IEEE J. Quantum Electron. 25, 2665 (1989).
[2] Agrawal, G.P., Nonlinear Fiber Optics, 6th ed. (Academic, 2019).
[3] Lin, Q. & Agrawal, G.P., Opt. Lett. 31, 3086 (2006).
[4] Hollenbeck, D. & Cantrell, C.D., J. Opt. Soc. Am. B 19, 2886 (2002).
"""

import warnings
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson

# ---------------------------------------------------------------------------
# Physical constants  (names match brillouin.py)
# ---------------------------------------------------------------------------

_HBAR = 1.054571817e-34   # J s
_KB   = 1.380649e-23      # J / K
_C    = 299792458.0        # m / s
_C_CM = 2.99792458e10     # cm / s  (for wavenumber conversions)
_PI   = np.pi

# ---------------------------------------------------------------------------
# Model selector
# ---------------------------------------------------------------------------

RAMAN_MODEL: str = 'bw'
"""Active Raman model.  One of ``'bw'``, ``'tabulated'``, ``'hc'``."""

VALID_MODELS = ('bw', 'tabulated', 'hc')


def set_model(model: str) -> None:
    """Set the module-level default Raman model."""
    global RAMAN_MODEL
    if model not in VALID_MODELS:
        raise ValueError(f"model must be one of {VALID_MODELS}, got {model!r}")
    RAMAN_MODEL = model


# ---------------------------------------------------------------------------
# Blow-Wood (BW) parameters
# ---------------------------------------------------------------------------

RAMAN_TAU1    = 12.2e-15    # s — phonon oscillation period (peak ~ 13.2 THz)
RAMAN_TAU2    = 32.0e-15    # s — phonon lifetime (sets linewidth)
RAMAN_FR      = 0.18        # fractional Raman contribution [2, Table 2.1]
RAMAN_OMEGA_R = 2 * _PI * 13.2e12   # rad/s — Raman peak frequency


# ═══════════════════════════════════════════════════════════════════════════
# Blow-Wood model
# ═══════════════════════════════════════════════════════════════════════════

def h_R_time(t):
    r"""Blow-Wood impulse response h_R(t).

    Causal (zero for t <= 0), normalised so that
    :math:`\int_0^\infty h_R(t)\,dt = 1`.

    Parameters
    ----------
    t : array_like
        Time (s).

    Returns
    -------
    ndarray
    """
    tau1, tau2 = RAMAN_TAU1, RAMAN_TAU2
    prefac = (tau1**2 + tau2**2) / (tau1 * tau2**2)
    return np.where(t > 0,
                    prefac * np.exp(-t / tau2) * np.sin(t / tau1),
                    0.0)


def h_R_freq(Omega):
    r"""Fourier transform of the BW impulse response.

    Convention: :math:`\tilde{h}_R(\Omega) = \int_0^\infty h_R(t)\,
    e^{+i\Omega t}\,dt` (Agrawal sign).
    :math:`\mathrm{Im}[\tilde{h}_R] > 0` for :math:`\Omega > 0`
    (Stokes gain).

    Parameters
    ----------
    Omega : array_like
        Angular frequency (rad/s).

    Returns
    -------
    ndarray, complex
    """
    tau1, tau2 = RAMAN_TAU1, RAMAN_TAU2
    num = (tau1**2 + tau2**2) / (tau1**2 * tau2**2)
    den = (1/tau2 - 1j * np.asarray(Omega, dtype=complex))**2 + (1/tau1)**2
    return num / den


# ═══════════════════════════════════════════════════════════════════════════
# Hollenbeck-Cantrell (HC) 13-mode model
# ═══════════════════════════════════════════════════════════════════════════
#
# Model equation (HC 2002, Eq. 9):
#   h_R(t) = sum_i A_i exp(-gamma_i t) exp(-Gamma_i^2 t^2 / 4) sin(w_i t)
#
# Unit conversions from wavenumbers (cm^-1):
#   w_i  [rad/s] = 2 pi c_cm * nu_i
#   Gamma [rad/s] = pi c_cm * Gaussian_FWHM
#   gamma [rad/s] = pi c_cm * Lorentzian_FWHM

# Table 1 of Hollenbeck & Cantrell (2002)
_HC_NU_CM = np.array([                                          # cm^-1
    56.25, 100.00, 231.25, 362.50, 463.00, 497.00, 611.50,
    691.67, 793.67, 835.50, 930.00, 1080.00, 1215.00])
_HC_A = np.array([                                              # A_i
    1.00, 11.40, 36.67, 67.67, 74.00, 4.50, 6.80,
    4.60, 4.20, 4.50, 2.70, 3.10, 3.00])
_HC_G_FWHM_CM = np.array([                                     # Gaussian FWHM
    52.10, 110.42, 175.00, 162.50, 135.33, 24.50, 41.50,
    155.00, 59.50, 64.30, 150.00, 91.00, 160.00])
_HC_L_FWHM_CM = np.array([                                     # Lorentzian FWHM
    17.37, 38.81, 58.33, 54.17, 45.11, 8.17, 13.83,
    51.67, 19.83, 21.43, 50.00, 30.33, 53.33])

# Convert to rad/s
_HC_OMEGA_V = 2.0 * _PI * _C_CM * _HC_NU_CM
_HC_GAMMA   = _PI * _C_CM * _HC_G_FWHM_CM
_HC_GAMMA_L = _PI * _C_CM * _HC_L_FWHM_CM


def _hc_h_R_unnorm(t):
    """Raw (unnormalised) HC impulse response."""
    t = np.asarray(t, dtype=float)
    h = np.zeros_like(t)
    mask = t > 0
    tp = t[mask]
    for A, wv, G, gL in zip(_HC_A, _HC_OMEGA_V, _HC_GAMMA, _HC_GAMMA_L):
        h[mask] += A * np.exp(-gL * tp) * np.exp(-G**2 * tp**2 / 4) * np.sin(wv * tp)
    return h


# Precompute HC spline at import time.
# Time grid:  0-3 ps in 0.5 fs steps (captures ~1 ps recurrence).
# Freq grid:  0-40 THz in 0.05 THz steps.
# Im[h_R(Omega)] = int_0^inf h_R(t) sin(Omega t) dt.

_HC_T_MAX     = 3.0e-12
_HC_DT        = 0.5e-15
_HC_OMEGA_MAX = 2.0 * _PI * 40.0e12
_HC_N_OMEGA   = 801

_t_hc      = np.arange(0.0, _HC_T_MAX, _HC_DT)
_h_hc_raw  = _hc_h_R_unnorm(_t_hc)
_HC_NORM   = np.trapezoid(_h_hc_raw, _t_hc)
_h_hc_norm = _h_hc_raw / _HC_NORM

_omega_hc_grid = np.linspace(0.0, _HC_OMEGA_MAX, _HC_N_OMEGA)
_sin_mat = np.sin(np.outer(_omega_hc_grid, _t_hc))
_im_hR_hc_pos = np.trapezoid(
    _h_hc_norm[np.newaxis, :] * _sin_mat, _t_hc, axis=1)
_HC_SPLINE = CubicSpline(_omega_hc_grid, _im_hR_hc_pos, extrapolate=False)

del _sin_mat  # free ~38 MB


def im_h_R_hc(Omega):
    r"""Im[h_R(Omega)] for the HC 13-mode model.

    Normalised so :math:`\int_0^\infty h_R(t)\,dt = 1`.
    Antisymmetric: :math:`\mathrm{Im}[\tilde{h}_R(-\Omega)]
    = -\mathrm{Im}[\tilde{h}_R(\Omega)]`.
    Zero for |Omega| / (2 pi) > 40 THz.

    Parameters
    ----------
    Omega : array_like
        Angular frequency (rad/s).

    Returns
    -------
    ndarray
    """
    Omega_arr = np.asarray(Omega, dtype=float)
    abs_Om = np.abs(Omega_arr)
    in_range = abs_Om <= _HC_OMEGA_MAX
    vals = np.nan_to_num(_HC_SPLINE(abs_Om), nan=0.0)
    vals = np.maximum(vals, 0.0)
    vals = np.where(in_range, vals, 0.0)
    return np.where(Omega_arr >= 0.0, vals, -vals)


def h_R_time_hc(t):
    """HC normalised impulse response in the time domain.

    Parameters
    ----------
    t : array_like
        Time (s).

    Returns
    -------
    ndarray
    """
    return _hc_h_R_unnorm(np.asarray(t, dtype=float)) / _HC_NORM


# ═══════════════════════════════════════════════════════════════════════════
# Unified gain dispatcher
# ═══════════════════════════════════════════════════════════════════════════

def g_R(Omega, gamma, fR=RAMAN_FR, model=None):
    r"""Raman gain coefficient g_R(Omega).

    Positive for Omega > 0 (Stokes gain), negative for Omega < 0
    (anti-Stokes absorption).

    .. math::
        g_R(\Omega) = 2\,\gamma\,f_R\,\mathrm{Im}[\tilde{h}_R(\Omega)]

    Parameters
    ----------
    Omega : array_like
        Angular frequency shift from pump (rad/s).
    gamma : float
        Fiber nonlinear coefficient (W^-1 m^-1).
    fR : float
        Raman fraction (default 0.18).
    model : {'bw', 'tabulated', 'hc'} or None
        Response model.  ``None`` reads ``RAMAN_MODEL``.

    Returns
    -------
    ndarray
        Gain coefficient (W^-1 m^-1).
    """
    m = model if model is not None else RAMAN_MODEL
    if m == 'bw':
        im_hR = np.imag(h_R_freq(Omega))
    elif m == 'tabulated':
        from raman_tabulated import im_h_R_tabulated
        im_hR = im_h_R_tabulated(Omega)
    elif m == 'hc':
        im_hR = im_h_R_hc(Omega)
    else:
        raise ValueError(f"Unknown model {m!r}. Choose from {VALID_MODELS}.")
    return 2.0 * gamma * fR * im_hR


def g_R_from_wavelengths(lambda_pump, lambda_signal, gamma,
                         fR=RAMAN_FR, model=None):
    """Evaluate g_R at the frequency shift between two wavelengths.

    Parameters
    ----------
    lambda_pump, lambda_signal : float
        Wavelengths (m).
    gamma : float
        Nonlinear coefficient (W^-1 m^-1).
    fR : float
        Raman fraction.
    model : str or None
        Response model.

    Returns
    -------
    float
        Gain coefficient (W^-1 m^-1).
    """
    Omega = 2.0 * _PI * _C * (1.0 / lambda_pump - 1.0 / lambda_signal)
    return float(g_R(Omega, gamma, fR=fR, model=model))


# ═══════════════════════════════════════════════════════════════════════════
# Thermal phonon occupancy
# ═══════════════════════════════════════════════════════════════════════════

def thermal_phonon_number(Omega, T_K):
    r"""Bose-Einstein mean phonon occupancy.

    .. math::
        n_\mathrm{th} = \frac{1}{\exp(\hbar|\Omega|/k_B T) - 1}

    Parameters
    ----------
    Omega : array_like
        Phonon angular frequency (rad/s); magnitude is used.
    T_K : float
        Temperature (K).

    Returns
    -------
    ndarray
    """
    x = _HBAR * np.abs(Omega) / (_KB * T_K)
    return np.where(x > 500.0, 0.0, 1.0 / np.expm1(x))


# Legacy alias
thermal_photon_number = thermal_phonon_number


# ═══════════════════════════════════════════════════════════════════════════
# Spontaneous Raman photon-rate density
# ═══════════════════════════════════════════════════════════════════════════

def _sprs_rate_density_core(gR, P_pump, L, factor, pump_depletion=False):
    """Core rate-density formula (shared by Stokes and anti-Stokes)."""
    if not pump_depletion:
        return P_pump * L * gR * factor / (2.0 * _PI)
    x = gR * P_pump * L
    corrected_PL = np.where(gR > 1e-25, np.expm1(x) / gR, P_pump * L)
    return corrected_PL * factor / (2.0 * _PI)


def sprs_photon_rate_density(Omega, omega_pump, P_pump, L, gamma, T_K,
                             sideband='stokes', pump_depletion=False,
                             model=None):
    r"""Spectral density of spontaneous Raman photons.

    Returns dN/dt/dOmega in photons s^-1 (rad/s)^-1.

    Parameters
    ----------
    Omega : array_like
        Signed frequency shift from pump (rad/s).
        Positive = Stokes, negative = anti-Stokes.
    omega_pump : float
        Pump angular frequency (rad/s).
    P_pump : float
        Pump power (W).
    L : float
        Effective fiber length (m).
    gamma : float
        Nonlinear coefficient (W^-1 m^-1).
    T_K : float
        Temperature (K).
    sideband : {'stokes', 'antistokes'}
    pump_depletion : bool
        Use the exponential correction when True.
    model : str or None

    Returns
    -------
    ndarray
        Photon rate spectral density (photons s^-1 (rad/s)^-1).
    """
    Omega_gR = np.abs(Omega) if sideband == 'antistokes' else Omega
    gR  = g_R(Omega_gR, gamma, model=model)
    nth = thermal_phonon_number(Omega, T_K)
    factor = (nth + 1.0) if sideband == 'stokes' else nth
    return np.maximum(
        _sprs_rate_density_core(gR, P_pump, L, factor, pump_depletion), 0.0)


# Legacy alias
spRam_photon_rate_density = sprs_photon_rate_density


def sprs_noise_in_channel(lambda_pump, lambda_channel, delta_lambda,
                          P_pump, L, gamma, T_K,
                          sideband='stokes', pump_depletion=False,
                          n_points=1001, model=None):
    """Total spontaneous Raman photon rate into a channel bandwidth.

    Integrates :func:`sprs_photon_rate_density` over a rectangular
    filter of width ``delta_lambda`` centred at ``lambda_channel``.

    Parameters
    ----------
    lambda_pump : float
        Pump wavelength (m).
    lambda_channel : float
        Channel centre wavelength (m).
    delta_lambda : float
        Channel filter bandwidth (m, full-width).
    P_pump : float
        Pump power (W).
    L : float
        Effective fiber length (m).
    gamma : float
        Nonlinear coefficient (W^-1 m^-1).
    T_K : float
        Temperature (K).
    sideband : {'stokes', 'antistokes'}
    pump_depletion : bool
    n_points : int
        Quadrature points (odd for Simpson).
    model : str or None

    Returns
    -------
    float
        Photon rate (photons/s).
    """
    omega_pump = 2.0 * _PI * _C / lambda_pump
    omega_ch   = 2.0 * _PI * _C / lambda_channel
    delta_omega = 2.0 * _PI * _C * delta_lambda / lambda_channel**2

    omega_arr = np.linspace(omega_ch - delta_omega / 2,
                            omega_ch + delta_omega / 2, n_points)
    Omega_arr = omega_pump - omega_arr

    density = sprs_photon_rate_density(
        Omega_arr, omega_pump, P_pump, L, gamma, T_K,
        sideband=sideband, pump_depletion=pump_depletion, model=model)

    return float(simpson(density, x=omega_arr))


# Legacy alias
spRam_noise_in_channel = sprs_noise_in_channel


# ═══════════════════════════════════════════════════════════════════════════
# Pump-depletion diagnostic
# ═══════════════════════════════════════════════════════════════════════════

def check_depletion_validity(P_pump, L, gamma, fR=RAMAN_FR, threshold=0.1):
    """Warn if the pump-depletion correction exceeds *threshold*.

    Returns the correction ratio ``[exp(x) - 1] / x`` where
    ``x = g_R_peak * P * L``.

    Parameters
    ----------
    P_pump : float
        Pump power (W).
    L : float
        Effective fiber length (m).
    gamma : float
        Nonlinear coefficient (W^-1 m^-1).
    fR : float
        Raman fraction.
    threshold : float
        Fractional deviation that triggers a warning (default 0.10).

    Returns
    -------
    float
        Correction ratio (1.0 = no correction needed).
    """
    gR_peak = g_R(RAMAN_OMEGA_R, gamma, fR=fR, model='bw')
    x = gR_peak * P_pump * L
    ratio = float(np.expm1(x) / x) if x > 1e-10 else 1.0
    if ratio - 1.0 > threshold:
        warnings.warn(
            f"Pump-depletion correction is {(ratio-1)*100:.1f}% "
            f"(x = gR P L = {x:.3f}). Pass pump_depletion=True.",
            UserWarning, stacklevel=2,
        )
    return ratio


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
        print('raman.py — self-tests')
        print('=' * 58)

    # BW impulse response normalisation
    t = np.linspace(0, 5e-12, 100_000)
    integral = float(np.trapezoid(h_R_time(t), t))
    check('BW h_R(t) integral ~ 1',
          abs(integral - 1.0) < 0.01,
          f'{integral:.4f}', '1.0')

    # BW gain is positive at Raman peak (Stokes)
    gR_peak = float(g_R(RAMAN_OMEGA_R, 1.3e-3, model='bw'))
    check('BW g_R(Omega_R) > 0 (Stokes)',
          gR_peak > 0, f'{gR_peak:.3e}', '> 0')

    # HC spline exists and is positive at 13 THz
    im_hc = float(im_h_R_hc(RAMAN_OMEGA_R))
    check('HC Im[h_R] > 0 at 13.2 THz',
          im_hc > 0, f'{im_hc:.4f}', '> 0')

    # HC normalisation
    integral_hc = float(np.trapezoid(h_R_time_hc(t), t))
    check('HC h_R(t) integral ~ 1',
          abs(integral_hc - 1.0) < 0.01,
          f'{integral_hc:.4f}', '1.0')

    # Thermal phonon number at Raman frequency
    nth = float(thermal_phonon_number(RAMAN_OMEGA_R, 300.0))
    check('n_th(13 THz, 300 K) ~ 0.14',
          0.05 < nth < 0.25,
          f'{nth:.3f}', '~0.14')

    # Noise rate is non-negative
    rate = sprs_noise_in_channel(
        1451e-9, 1550e-9, 1e-9, 1e-3, 25e3, 1.3e-3, 300.0, model='bw')
    check('SpRS noise >= 0',
          rate >= 0, f'{rate:.2e}', '>= 0')

    # Legacy aliases exist
    check('Legacy aliases present',
          callable(spRam_noise_in_channel)
          and callable(spRam_photon_rate_density)
          and callable(thermal_photon_number))

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

    gamma = 1.3e-3  # W^-1 m^-1
    print()
    print('Summary  (gamma = 1.3e-3 W^-1 m^-1, 300 K)')
    print('-' * 48)
    for m in VALID_MODELS:
        try:
            gR_peak = float(g_R(RAMAN_OMEGA_R, gamma, model=m))
            print(f'  {m:>10}  g_R(13.2 THz) = {gR_peak:.4e} W^-1 m^-1')
        except ImportError:
            print(f'  {m:>10}  (not available)')
    print(f'  n_th(13 THz, 300 K) = {thermal_phonon_number(RAMAN_OMEGA_R, 300.):.4f}')
